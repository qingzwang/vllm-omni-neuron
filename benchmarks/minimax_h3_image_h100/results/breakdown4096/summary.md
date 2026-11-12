| 阶段 | vLLM-Omni TP2 | SGLang Ulysses2 | SGLang TP2 |
|---|---:|---:|---:|
| ① 请求接入 (HTTP→API→调度→GPU worker) | 4 ms | 5 ms | 4 ms |
| ② 请求解析 + 噪声/latent 准备 | 15 ms | 7 ms | 9 ms |
| ③ 文本编码器 Qwen3-VL | 68 ms | 77 ms | 82 ms |
| ④ DiT 去噪前准备 (packing/RoPE/AdaLN) | 2 ms | 6 ms | 31 ms |
| ⑤ DiT 去噪 6 步 合计 | 5299 ms | 4947 ms | 5226 ms |
| 　每步 (step0 / step1-5 均值) | 859 / 888 ms | 797 / 830 ms | 848 / 876 ms |
| ⑥ VAE 解码 (+uint8 量化, GPU→CPU) | 587 ms | 282 ms | 279 ms |
| ⑦ pipeline 其他开销 | — | 22 ms | 29 ms |
| ⑧ PNG 编码 | 166 ms (PNG+b64) | 249 ms | 252 ms |
| ⑨ worker→API 进程 IPC 传输 | 115 ms | 185 ms | 189 ms |
| ⑩ base64 编码 | (含在⑧) | 123 ms | 121 ms |
| ⑪ JSON 序列化 + 发送 | 39 ms | 176 ms | 173 ms |
| ⑫ 客户端下载 67MB 响应体 | 77 ms | 77 ms | 76 ms |
| ⑬ 客户端 JSON 解析 + base64 解码 | 225 ms | 221 ms | 220 ms |
| **服务端 GPU 计算 (②–⑥)** | **5972 ms** | **5318 ms** | **5627 ms** |
| **E2E 总延迟 (到客户端拿到图片字节)** | **6.597 s** | **6.387 s** | **6.676 s** |
