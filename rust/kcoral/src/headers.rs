use http::HeaderMap;

const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

pub fn forwarded(headers: &HeaderMap, extra_excluded: &[&str]) -> HeaderMap {
    let mut connection_tokens = Vec::new();
    for value in headers.get_all(http::header::CONNECTION) {
        if let Ok(value) = value.to_str() {
            connection_tokens.extend(
                value
                    .split(',')
                    .map(|token| token.trim().to_ascii_lowercase()),
            );
        }
    }

    let mut output = HeaderMap::new();
    for (name, value) in headers {
        let lower = name.as_str();
        if HOP_BY_HOP.contains(&lower)
            || connection_tokens.iter().any(|token| token == lower)
            || extra_excluded.contains(&lower)
        {
            continue;
        }
        output.append(name.clone(), value.clone());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;

    #[test]
    fn removes_hop_by_hop_and_connection_named_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "connection",
            HeaderValue::from_static("x-private, keep-alive"),
        );
        headers.insert("x-private", HeaderValue::from_static("secret"));
        headers.insert("x-kept", HeaderValue::from_static("yes"));
        let forwarded = forwarded(&headers, &[]);
        assert_eq!(forwarded.len(), 1);
        assert_eq!(forwarded["x-kept"], "yes");
    }

    #[test]
    fn preserves_repeated_headers() {
        let mut headers = HeaderMap::new();
        headers.append("set-cookie", HeaderValue::from_static("a=1"));
        headers.append("set-cookie", HeaderValue::from_static("b=2"));
        let forwarded = forwarded(&headers, &[]);
        assert_eq!(forwarded.get_all("set-cookie").iter().count(), 2);
    }
}
