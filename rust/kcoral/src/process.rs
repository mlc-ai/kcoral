//! Linux process ownership for one node manager and one managed server tree.
//!
//! Workers create their own sessions. A process group alone cannot contain them.
//! The dedicated node binary adopts orphan descendants and signals live process
//! handles, so PID reuse cannot redirect a signal to an unrelated process.
use nix::libc;
use std::{
    collections::HashSet,
    fs, io,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    sync::Arc,
};

/// Enable orphan adoption before starting any server. The caller must be a
/// dedicated node process: all its descendants belong to the managed server.
pub fn adopt_server_descendants() -> io::Result<()> {
    if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } == -1 {
        return Err(io::Error::last_os_error());
    }
    // Check kernel/container support before launching a server tree.
    let handle = open_process(std::process::id())?;
    signal_process(&handle, 0)?;
    Ok(())
}

fn open_process(pid: u32) -> io::Result<OwnedFd> {
    let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
    if raw == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(raw as i32) })
}

fn signal_process(handle: &OwnedFd, signal: i32) -> io::Result<bool> {
    let result = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            handle.as_raw_fd(),
            signal,
            std::ptr::null::<libc::siginfo_t>(),
            0,
        )
    };
    if result == -1 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(false);
        }
        return Err(error);
    }
    Ok(true)
}

pub(crate) struct ProcessTree {
    root: u32,
    root_handle: Arc<OwnedFd>,
    adopt: bool,
}

impl ProcessTree {
    pub(crate) fn new(root: u32) -> io::Result<Self> {
        let mut enabled: libc::c_int = 0;
        unsafe {
            libc::prctl(libc::PR_GET_CHILD_SUBREAPER, &mut enabled, 0, 0, 0);
        }
        Ok(Self {
            root,
            root_handle: Arc::new(open_process(root)?),
            adopt: enabled != 0,
        })
    }

    fn descendants(&self) -> io::Result<Vec<(u32, Arc<OwnedFd>)>> {
        let owner = if self.adopt {
            std::process::id()
        } else {
            self.root
        };
        let parent_handle = if self.adopt {
            None
        } else {
            Some(self.root_handle.clone())
        };
        let mut pending = vec![(owner, parent_handle)];
        let mut seen = HashSet::from([owner]);
        let mut result = Vec::new();
        while let Some((parent, parent_handle)) = pending.pop() {
            let tasks = match fs::read_dir(format!("/proc/{parent}/task")) {
                Ok(tasks) => tasks,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            };
            for task in tasks {
                let path = task?.path().join("children");
                let children = match fs::read_to_string(path) {
                    Ok(children) => children,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(e),
                };
                for pid in children
                    .split_whitespace()
                    .filter_map(|s| s.parse::<u32>().ok())
                {
                    if !seen.insert(pid) {
                        continue;
                    }
                    let handle = match open_process(pid) {
                        Ok(handle) => Arc::new(handle),
                        Err(error) if error.raw_os_error() == Some(libc::ESRCH) => continue,
                        Err(error) => return Err(error),
                    };
                    // Validate parentage after opening the handle. If it changed,
                    // the next scan rediscovers an adopted child from its new parent.
                    let stat = match fs::read_to_string(format!("/proc/{pid}/stat")) {
                        Ok(stat) => stat,
                        Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                        Err(e) => return Err(e),
                    };
                    let ppid = stat
                        .rsplit_once(") ")
                        .and_then(|(_, fields)| fields.split_whitespace().nth(1))
                        .and_then(|s| s.parse::<u32>().ok());
                    if ppid != Some(parent) {
                        continue;
                    }
                    // The parent must still be the process whose children we
                    // enumerated. A recycled parent PID could otherwise lead us
                    // into an unrelated tree despite using stable child handles.
                    if let Some(parent_handle) = &parent_handle {
                        if !signal_process(parent_handle, 0)? {
                            continue;
                        }
                    }
                    pending.push((pid, Some(handle.clone())));
                    result.push((pid, handle));
                }
            }
        }
        Ok(result)
    }

    /// Signal descendants individually, including workers that called setsid.
    /// Repeated scans find children born during a previous scan or reparenting.
    pub(crate) fn signal(&self, signal: i32) -> io::Result<bool> {
        let children = self.descendants()?;
        let present = !children.is_empty();
        for (pid, handle) in children.into_iter().rev() {
            signal_process(&handle, signal)?;
            // Child::wait owns the root; only reap orphan descendants here.
            if self.adopt && pid != self.root {
                let mut status = 0;
                unsafe {
                    libc::waitpid(pid as i32, &mut status, libc::WNOHANG);
                }
            }
        }
        Ok(present)
    }

    pub(crate) async fn finish(&self) -> anyhow::Result<()> {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if !self.signal(libc::SIGKILL)? {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                anyhow::bail!(
                    "server descendants did not exit; refusing to restart alongside them"
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

impl Drop for ProcessTree {
    fn drop(&mut self) {
        let _ = self.signal(libc::SIGKILL);
    }
}
