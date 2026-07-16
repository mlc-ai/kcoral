# GPU Server v2 API 参考

## 1. 概述

GPU Server v2 是一个同步远程 GPU 执行服务。GPU 指图形处理器，API 指应用程序编程接口。

客户端在一次请求中上传：

- 一个入口脚本，默认路径为 `main.py`；
- 零个或多个脚本运行所需的工件文件；
- 可选的执行设置，包括语言、入口位置和超时时间。

这里的“工件”指配置文件、输入数据、Python 模块、动态链接库以及其他供入口脚本读取的文件。

服务端为请求创建独立工作目录，加载入口文件，调用其中无参数的入口函数，然后把函数返回值传回调用方。入口文件和入口函数默认为 `main.py` 中的 `main()`。

协议使用 `language` 字段描述入口脚本的语言。第一版只支持 Python，协议形状为后续语言预留。

协议中没有指令列表、操作码、寄存器、远程函数名和输出文件声明。模块加载、张量构造、GPU 内核执行、正确性检查和性能测量都由上传的脚本完成。

第一版只面向可信客户端。上传的代码可以使用工作进程权限执行任意操作，该接口不提供安全沙箱。

---

## 2. 接口

### `POST /execute`

上传并在一张 GPU 上执行一个程序。

该请求采用同步模式。请求在排队和执行期间保持连接，成功响应携带入口函数的返回值。

### `GET /health`

返回服务状态、GPU 工作进程状态和排队请求数量。

---

## 3. `POST /execute`

### 3.1 请求格式

内容类型：

```text
multipart/form-data
```

`multipart/form-data` 是一种 HTTP 多段请求格式，可以在一个请求体中携带 JSON 元数据和多个二进制文件。HTTP 指超文本传输协议，JSON 指 JavaScript Object Notation，一种结构化文本数据格式。

请求体支持以下部分：

| 部分名称 | 内容类型 | 必需 | 说明 |
|---|---|---:|---|
| `job` | `application/json` | 否 | 执行设置 |
| `file:<path>` | `application/octet-stream` | 是 | 放到工作目录 `<path>` 的文件 |

上传文件中必须恰好有一个文件的路径等于入口文件路径（默认 `main.py`）。

工件部分名称示例：

```text
file:input.json
file:data/input.bin
file:modules/kernel.py
file:build/kernel.so
```

`file:` 前缀属于协议元数据，创建文件时会被移除。例如，`file:data/input.bin` 会成为：

```text
<工作目录>/data/input.bin
```

服务端为每个文件计算完整的 SHA-256 内容哈希。SHA-256 是一种密码学哈希函数，这里用来识别内容相同的文件。服务端可以在内部复用缓存文件；缓存不会改变请求语义，每个文件仍会出现在本次请求的工作目录中。

### 3.2 `job` 对象

可选的 `job` 部分格式如下：

```json
{
  "language": "python",
  "entry": {
    "file": "main.py",
    "function": "main"
  },
  "timeout_seconds": 60
}
```

| 字段 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `language` | 否 | `"python"` | 入口脚本的语言 |
| `entry` | 否 | 见下 | 入口位置 |
| `entry.file` | 否 | `"main.py"` | 入口文件在工作目录中的路径 |
| `entry.function` | 否 | `"main"` | 入口函数名 |
| `timeout_seconds` | 否 | 服务端配置 | 正数，表示执行超时时间，单位为秒 |

所有字段都可以省略。省略 `job` 等价于全部使用默认值，此时服务端执行：

```python
from main import main

return_value = main()
```

`language` 第一版只接受 `"python"`。其他取值返回 `invalid_request`。该字段决定服务端如何加载入口文件和调用入口函数；后续语言必须在协议修订中定义各自的加载和调用规则。

`entry.file` 必须满足 3.3 节的路径规则，并且必须指向本次请求上传的一个文件。`entry.function` 必须是入口文件中定义的无参数可调用函数。

服务端拒绝未知字段，避免客户端误以为某项未支持的设置已经生效。

`job` 不声明输出。应用层结果只有入口函数的返回值。

文档其余部分使用默认入口 `main.py` 和 `main()` 描述行为；除非特别说明，这些描述对自定义 `entry` 同样成立，把 `main.py` 替换为 `entry.file`、`main()` 替换为 `entry.function` 即可。

### 3.3 文件路径规则

所有工件路径必须满足以下条件：

- 使用 `/` 作为路径分隔符；
- 使用相对于工作目录的路径；
- 不包含空路径段、`.` 或 `..`；
- 不包含空字符；
- 不以 `/` 开头；
- 不允许通过符号链接指向工作目录之外；
- 在一次请求中保持唯一；
- 不覆盖服务端创建的内部文件。

路径规范化之后必须与客户端提交的路径完全相同：

| 客户端提交的部分 | 处理结果 |
|---|---|
| `file:data/input.bin` | 接受 |
| `file:./input.bin` | 拒绝 |
| `file:data/../input.bin` | 拒绝 |
| `file:/etc/passwd` | 拒绝 |

服务端按需创建父目录。

### 3.4 Python 入口约定

`language` 为 `"python"` 时，入口文件必须定义一个与 `entry.function` 同名的可调用函数。使用默认值时即：

```python
def main():
    ...
    return result
```

入口函数必须满足：

- 不接收参数；
- 可以导入同一请求上传的其他 Python 文件；
- 可以使用相对于当前工作目录的路径读取工件；
- 运行时只能看到一张 GPU，该设备显示为 `cuda:0`；
- 返回一个受支持的值；
- 在请求超时前完成。

服务端在导入入口文件之前，把当前工作目录切换到请求工作目录，并在请求期间将该目录放到 Python 模块搜索路径的最前面。

服务端以模块方式导入入口文件。文件顶层代码会在调用入口函数之前执行。建议把实际工作放在入口函数内，便于确定失败位置和测量执行时间。

服务端不传递命令行参数，不注入应用对象，也不查找 `entry.function` 以外的函数名。

### 3.5 脚本执行环境

每张 GPU 对应一个工作槽位，同一槽位上的请求串行执行。

服务端在执行脚本前限制 GPU 可见范围，使分配到的物理 GPU 在脚本中显示为：

```text
cuda:0
```

脚本使用服务端预先配置的 Python 解释器和已安装依赖。客户端不能通过该接口选择其他解释器或安装依赖。

服务端提供以下环境变量：

| 环境变量 | 说明 |
|---|---|
| `GPU_SERVER_REQUEST_ID` | 用于日志和问题定位的唯一请求标识 |
| `GPU_SERVER_WORK_DIR` | 本次请求工作目录的绝对路径 |

程序应优先使用相对路径访问上传的工件。工作目录绝对路径仅对当前请求有效，不应持久化使用。

### 3.6 请求标识

服务端在 `/execute` 请求进入处理函数后、解析请求体之前生成一个 UUID v4。UUID 指通用唯一标识符，v4 表示该标识符使用随机数生成。

请求标识使用小写标准 UUID 字符串：

```text
7f61b94e-034a-4e80-b67d-eca52bb952cc
```

该值在协议中统一命名为 `request_id`，并贯穿请求解析、排队、工作进程执行、日志记录和响应生成。

每个 `/execute` 响应都通过 HTTP 响应头返回请求标识：

```http
X-Request-ID: 7f61b94e-034a-4e80-b67d-eca52bb952cc
```

成功和错误响应的 JSON 元数据也包含相同的 `request_id`。对于字节串和张量返回值，该字段位于 multipart 响应的 `result` 部分。

客户端不能指定或覆盖 `request_id`。同一次 HTTP 请求在服务端内部始终使用同一个值；客户端重试会产生新的 `request_id`。如果以后需要识别业务层重复提交，应增加独立的幂等键，不能使用 `request_id` 表示幂等关系。

`GPU_SERVER_REQUEST_ID` 环境变量的值与响应中的 `request_id` 完全一致。

---

## 4. 返回值

`main()` 可以返回由以下类型递归组成的值：

- `None`、`bool`、`int`、有限的 `float` 和 `str`；
- `bytes`、`bytearray` 和 `memoryview`；
- 张量；
- `list`；
- `tuple`；
- 键为字符串的 `dict`。

列表、元组和字典可以在任意层级包含上述类型。一个返回值可以同时包含多个字节串和多个张量。

张量指具有数据类型、形状和数值数据的多维数组。服务端通过运行环境提供的张量适配器读取张量，将其转换为连续的主机内存字节。

### 4.1 返回值描述树

服务端将 Python 返回值递归编码为一棵描述树：

| Python 值 | 描述节点 |
|---|---|
| 完全可表示为 JSON 的子树 | `{"type": "json", "value": ...}` |
| `bytes`、`bytearray`、`memoryview` | `{"type": "bytes", ...}` |
| 张量 | `{"type": "tensor", ...}` |
| `list` | `{"type": "list", "items": [...]}` |
| `tuple` | `{"type": "tuple", "items": [...]}` |
| `dict` | `{"type": "dict", "items": {...}}` |

JSON 节点可以包含：

- `null`；
- 布尔值；
- 整数；
- 有限浮点数；
- 字符串；
- 只包含 JSON 值的数组；
- 键为字符串、值为 JSON 值的对象。

元组在描述树中保留为 `tuple`，客户端可以据此恢复元组。正无穷、负无穷和非数值等非有限浮点数不属于 JSON 值。

如果一个完整子树都可以表示为 JSON，服务端将该子树合并为一个 `json` 节点，避免为每个标量生成描述节点。

### 4.2 纯 JSON 返回值

返回值完全由 JSON 值组成时，响应内容类型为 `application/json`。

`main.py` 示例：

```python
def main():
    return {
        "correct": True,
        "median_ms": 0.128,
        "samples_ms": [0.127, 0.128, 0.131],
    }
```

成功响应：

```json
{
  "status": "ok",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "return": {
    "type": "json",
    "value": {
      "correct": true,
      "median_ms": 0.128,
      "samples_ms": [0.127, 0.128, 0.131]
    }
  },
  "elapsed_ms": 1245.6,
  "queue_ms": 18.2,
  "stdout": "",
  "stderr": ""
}
```

### 4.3 包含二进制值的返回值

返回值树中出现字节串或张量时，响应内容类型为 `multipart/form-data`，包含：

| 部分名称 | 内容类型 | 说明 |
|---|---|---|
| `result` | `application/json` | 执行元数据和完整返回值描述树 |
| `return:<index>` | `application/octet-stream` | 一个字节串或张量的原始字节 |

`<index>` 从 `0` 开始。服务端按深度优先遍历顺序为二进制值分配 part 标识。part 标识在 JSON 中使用字符串表示，例如：

```json
"part": "return:0"
```

该字符串与 multipart 部分的 `name` 完全相同：

```http
Content-Disposition: form-data; name="return:0"
Content-Type: application/octet-stream
```

客户端必须使用描述节点中的 `part` 字段查找数据，不应自行推算编号或解析标识符中的数字。part 标识只在当前 HTTP 响应中有效。

每个二进制节点同时携带：

| 字段 | 说明 |
|---|---|
| `part` | 当前响应中的 multipart 部分名称 |
| `size` | 原始数据字节数 |
| `sha256` | 原始数据的完整 SHA-256 哈希 |

`part` 用于定位数据，`sha256` 用于完整性校验。多个返回值节点可以引用同一个 part，以复用内容完全相同的二进制数据。

### 4.4 字节串节点

字节串节点格式：

```json
{
  "type": "bytes",
  "part": "return:0",
  "size": 15,
  "sha256": "<sha256>"
}
```

对应的 multipart 部分保存 `bytes`、`bytearray` 或 `memoryview` 的原始字节。

### 4.5 张量节点

张量节点格式：

```json
{
  "type": "tensor",
  "dtype": "float16",
  "shape": [32, 128],
  "part": "return:0",
  "size": 8192,
  "sha256": "<sha256>"
}
```

服务端对张量执行设备同步，将张量转换为连续布局并复制到主机内存。张量字节使用连续的行优先顺序存储，多字节标量使用小端字节序。

张量节点中的：

- `dtype` 表示元素数据类型；
- `shape` 表示各维长度；
- `size` 必须等于形状中各维长度的乘积乘以单个元素的字节数。

### 4.6 嵌套返回值示例

`main()` 可以返回：

```python
def main():
    return {
        "correct": True,
        "metrics": {
            "median_ms": 0.128,
            "samples_ms": [0.127, 0.128, 0.131],
        },
        "outputs": [
            output_tensor,
            b"binary metadata",
        ],
    }
```

`result` 部分中的返回值描述树为：

```json
{
  "status": "ok",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "return": {
    "type": "dict",
    "items": {
      "correct": {
        "type": "json",
        "value": true
      },
      "metrics": {
        "type": "json",
        "value": {
          "median_ms": 0.128,
          "samples_ms": [0.127, 0.128, 0.131]
        }
      },
      "outputs": {
        "type": "list",
        "items": [
          {
            "type": "tensor",
            "dtype": "float16",
            "shape": [32, 128],
            "part": "return:0",
            "size": 8192,
            "sha256": "<sha256>"
          },
          {
            "type": "bytes",
            "part": "return:1",
            "size": 15,
            "sha256": "<sha256>"
          }
        ]
      }
    }
  },
  "elapsed_ms": 1245.6,
  "queue_ms": 18.2,
  "stdout": "",
  "stderr": ""
}
```

multipart 响应还包含 `return:0` 和 `return:1` 两个二进制部分。

### 4.7 类型和序列化限制

返回值不属于第 4 节列出的类型时，服务端返回 `unsupported_return_type`。例如：

- 生成器和迭代器；
- 打开的文件对象；
- 任意其他 Python 类实例；
- 函数和模块；
- 键不是字符串的字典。

服务端应在错误信息中提供无法编码的值路径，例如：

```json
{
  "status": "error",
  "error": "unsupported_return_type",
  "message": "unsupported value at $.outputs[2].metadata",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc"
}
```

服务端不保留 Python 对象引用关系。同一个对象在返回值树中出现多次时，客户端得到多个独立引用；对应二进制内容可以共享同一个 part。

循环引用无法表示为有限描述树。服务端在递归构建描述树时必须主动检查循环引用，不能等待 Python 达到递归深度限制。

检查时记录当前递归路径上 `list`、`tuple` 和 `dict` 的对象标识：

1. 进入容器前，如果它的对象标识已经位于当前递归路径中，则发现循环引用；
2. 进入容器时，将对象标识加入当前递归路径；
3. 完成该容器编码后，将对象标识移出当前递归路径。

该集合只记录当前递归路径，不记录所有已经访问的对象。因此，多个位置可以引用同一个非循环对象；这些位置会分别编码。

发现循环引用时，服务端返回 `return_serialization_failed`，并在错误信息中携带发现循环的位置：

```json
{
  "status": "error",
  "error": "return_serialization_failed",
  "message": "circular reference at $.outputs[1]",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc"
}
```

描述树构建完成后，服务端使用 Python `json.dumps` 生成 `result`。调用时保持默认的 `check_circular=True`，并设置 `allow_nan=False`，对循环引用和非有限浮点数再做一次校验。自定义递归检查仍然必需，因为循环可能在描述树构建完成之前发生。

服务端必须配置：

- 最大嵌套深度；
- 最大描述节点数量；
- 最大 JSON 元数据字节数；
- 单个二进制值最大字节数；
- 整个响应最大字节数。

服务端在发送 HTTP 响应头之前完成返回值遍历，并将所有二进制内容序列化到临时文件。这样可以在遍历、同步或序列化失败时返回完整的 JSON 错误响应。

---

## 5. 成功响应元数据

每个成功响应都包含：

| 字段 | 说明 |
|---|---|
| `status` | 固定为 `"ok"` |
| `request_id` | 服务端为本次 HTTP 请求生成的 UUID v4 |
| `return` | `main()` 返回值的类型和表示 |
| `elapsed_ms` | 从开始导入 `main.py` 到返回值序列化完成的时间 |
| `queue_ms` | 等待可用 GPU 工作进程的时间 |
| `stdout` | 导入和执行 `main()` 期间捕获的标准输出 |
| `stderr` | 导入和执行 `main()` 期间捕获的标准错误 |

`elapsed_ms` 包含：

- 导入 `main.py`；
- 执行文件顶层代码；
- 执行 `main()`；
- 同步并序列化返回值。

`elapsed_ms` 不包含：

- HTTP 请求上传时间；
- 排队时间；
- HTTP 响应下载时间。

GPU 内核性能应由 `main()` 使用适合该运行环境的 GPU 同步和计时方式测量。`elapsed_ms` 描述服务端执行总耗时，不能作为内核性能结果。

服务端可以按配置限制标准输出和标准错误的最大字节数。发生截断时，响应额外包含：

```json
{
  "stdout_truncated": true,
  "stderr_truncated": false
}
```

---

## 6. 错误响应

所有错误响应都使用 `application/json`。

通用格式：

```json
{
  "status": "error",
  "error": "<error-type>",
  "message": "<human-readable message>",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "stdout": "",
  "stderr": ""
}
```

错误响应中的 `request_id` 是小写标准 UUID v4，与 `X-Request-ID` 响应头完全一致。如果脚本已经开始执行，响应会包含 `stdout` 和 `stderr`。

### 错误类型

| 错误 | HTTP 状态码 | 说明 |
|---|---:|---|
| `invalid_request` | 400 | 多段请求体或 `job` 对象无效 |
| `invalid_path` | 400 | 工件路径不安全、无效或重复 |
| `missing_entry` | 400 | 上传文件中没有 `entry.file` 指定的入口文件（默认 `main.py`） |
| `invalid_entry` | 400 | 无法导入入口文件，或文件没有定义与 `entry.function` 同名的可调用函数 |
| `execution_failed` | 400 | 导入阶段或入口函数抛出异常 |
| `unsupported_return_type` | 400 | `main()` 返回协议无法序列化的值 |
| `return_serialization_failed` | 500 | 同步或序列化受支持的返回值失败 |
| `timeout` | 408 | 执行时间超过 `timeout_seconds` |
| `worker_unavailable` | 503 | 没有健康的 GPU 工作进程可以接收请求 |
| `worker_crashed` | 503 | 工作进程退出且没有返回有效响应 |

执行失败响应包含 Python 调用栈：

```json
{
  "status": "error",
  "error": "execution_failed",
  "message": "main() raised RuntimeError: correctness check failed",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "traceback": "...",
  "stdout": "...",
  "stderr": "..."
}
```

调用栈可能泄露源代码和本地路径。面向不可信调用方的部署应关闭调用栈详情或对其进行清理。

---

## 7. 执行模型

### 7.1 进程模型

服务端包含：

- 一个接收 HTTP 请求并调度任务的前端进程；
- 每张已配置 GPU 对应一个工作槽位；
- 每张 GPU 同一时间最多执行一个请求。

前端进程不初始化 GPU 运行环境。每个工作进程通过 CUDA 设备可见性设置绑定到一张物理 GPU。CUDA 是工作环境使用的 GPU 编程和运行平台。

### 7.2 请求生命周期

每个请求按以下步骤执行：

1. 生成请求 UUID，并建立请求日志上下文。
2. 解析并验证多段请求体。
3. 验证 `job`、所有上传路径以及入口文件是否存在。
4. 计算文件内容哈希并获取缓存文件引用。
5. 等待一个空闲 GPU 工作进程。
6. 创建新的请求工作目录。
7. 在工作目录中放置所有上传文件。
8. 设置当前工作目录和 Python 模块搜索路径。
9. 导入入口文件。
10. 获取并验证入口函数。
11. 无参数调用入口函数。
12. 同步并序列化返回值。
13. 捕获标准输出和标准错误。
14. 删除工作目录并释放缓存文件引用。
15. 在响应头和响应元数据中返回请求 UUID。

成功、脚本失败、序列化失败和超时都会执行清理流程。

### 7.3 调度

前端维护先进先出的请求队列。多个 GPU 工作进程同时空闲时，前端按轮转顺序选择下一个工作进程。

每个工作进程串行执行请求，防止多个基准测试请求同时共享一张 GPU，减少性能测量受到的干扰。

`queue_ms` 统计从请求验证完成到分配工作进程之间的时间。

### 7.4 隔离和状态生命周期

以下状态只属于一个请求：

- 工作目录；
- 上传文件布局；
- 导入的入口模块；
- 捕获的标准输出和标准错误；
- 返回的 Python 对象；
- 只能从本次请求对象访问的 GPU 内存。

服务端不提供会话、跨请求 Python 对象、函数句柄、寄存器或应用层共享状态。

文件内容缓存可以跨请求存活。缓存只保存不可变的上传字节，不保存 Python 模块、GPU 张量或执行结果。

Python 和原生动态链接库可能创建进程级全局状态。实现必须保证请求无法观察到上一个请求导入的入口模块或工作目录。如果长期运行的工作进程无法彻底清理这些状态，在执行下一个请求前必须替换脚本运行进程。

### 7.5 超时与恢复

超时范围包含：

- 导入 `main.py`；
- 执行文件顶层代码；
- 执行 `main()`；
- 同步并序列化返回值。

排队时间不计入执行超时。

超时发生后，服务端执行：

1. 终止正在执行上传脚本的进程；
2. 等待一段较短且可配置的退出宽限时间；
3. 如果进程仍未退出，则强制结束；
4. 在该 GPU 接收下一个请求前替换受影响的运行进程；
5. 删除本次请求工作目录；
6. 返回 `timeout` 错误。

服务端不尝试恢复已经超时的 Python 代码。

---

## 8. 文件缓存

文件缓存只用于优化传输和工作目录构建。

缓存键是文件内容的完整 SHA-256 哈希，缓存条目不可修改。两个请求上传完全相同的字节时，可以引用同一缓存文件，同时在各自工作目录中使用不同路径。

文件缓存：

- 不改变 `main.py` 可见的 Python 接口；
- 不保留脚本对工作目录中文件的修改；
- 不缓存 `main()` 返回值；
- 不缓存已导入的 Python 模块；
- 不缓存 GPU 内存。

工作目录中的文件必须采用副本、只读链接或具有同等隔离效果的私有视图，防止一个请求修改另一个请求观察到的字节。

缓存驱逐只能删除没有活跃请求引用的条目。

---

## 9. `GET /health`

成功响应：

```json
{
  "status": "ok",
  "gpu_count": 2,
  "queue_length": 3,
  "workers": [
    {
      "gpu_id": 0,
      "status": "busy",
      "uptime_seconds": 3600
    },
    {
      "gpu_id": 1,
      "status": "idle",
      "uptime_seconds": 3580
    }
  ]
}
```

工作进程状态取值：

| 状态 | 说明 |
|---|---|
| `idle` | 可以接收请求 |
| `busy` | 正在执行请求 |
| `restarting` | 正在替换超时或失败的运行进程 |
| `unhealthy` | 无法执行请求 |

健康检查用于报告服务是否可接收请求，每次调用不会额外执行 GPU 内核。

---

## 10. 完整示例

### 上传的 `main.py`

```python
import json
from pathlib import Path


def main():
    config = json.loads(Path("config.json").read_text())
    input_data = Path("data/input.bin").read_bytes()

    # Build and run the GPU workload here.
    output_size = len(input_data)

    return {
        "correct": True,
        "output_size": output_size,
        "warmup_iterations": config["warmup_iterations"],
        "median_ms": 0.128,
    }
```

### 请求

```bash
curl -X POST http://server:8000/execute \
  -F 'job={"timeout_seconds":60};type=application/json' \
  -F 'file:main.py=@main.py;type=application/octet-stream' \
  -F 'file:config.json=@config.json;type=application/octet-stream' \
  -F 'file:data/input.bin=@input.bin;type=application/octet-stream'
```

### 响应

```json
{
  "status": "ok",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "return": {
    "type": "json",
    "value": {
      "correct": true,
      "output_size": 1048576,
      "warmup_iterations": 10,
      "median_ms": 0.128
    }
  },
  "elapsed_ms": 842.7,
  "queue_ms": 0.4,
  "stdout": "",
  "stderr": ""
}
```
