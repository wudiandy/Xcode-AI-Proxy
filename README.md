# Xcode AI Proxy

解决 Xcode 中无法直接添加智谱 GLM、Kimi 和 DeepSeek 模型的问题。

## 解决什么问题？

当你在 Xcode 中尝试添加智谱、Kimi 或 DeepSeek AI 提供商时，会遇到：

- ❌ "Provider is not valid"
- ❌ "Models could not be fetched with the provided account details"

这个代理服务让你可以在 Xcode 中正常使用智谱 GLM-4.5、Kimi 和 DeepSeek 模型。

## 使用方法

确保你已安装 Python 3.8+

#### 1. 配置 API 密钥

复制 `.env.example` 为 `.env`，填入你的 API 密钥：

```bash
# 智谱AI API 密钥 (从 https://open.bigmodel.cn/ 获取)
ZHIPU_API_KEY=你的智谱API密钥

# Kimi API 密钥 (从 https://platform.moonshot.cn/ 获取)
KIMI_API_KEY=你的Kimi API密钥

# DeepSeek API 密钥 (从 https://platform.deepseek.com/ 获取)
DEEPSEEK_API_KEY=你的DeepSeek API密钥
```

#### 2. 启动服务

**使用启动脚本（推荐）:**

```bash
./start_python.sh
```

**手动启动:**

```bash
pip3 install -r requirements.txt
python3 server_python.py
```

服务启动在 `http://localhost:3000`

### 3. 配置 Xcode

#### 3.1 在 Internet Hosted 中添加 AI 提供商：

- **Base URL**: `http://localhost:3000`
- **API Key**: `any-string-works` (任意字符串)

#### 3.2 在 Locally Hosted 中添加 端口：

- **端口**: `3000`

现在可以在 Xcode 中正常使用智谱 GLM-4.5、Kimi 和 DeepSeek 模型了！

## 支持的模型

- `glm-4.5` - 智谱 AI GLM-4.5
- `kimi-k2-0905-preview` - Kimi K2
- `deepseek-reasoner` - DeepSeek Reasoner (思维模式)
- `deepseek-chat` - DeepSeek Chat (对话模式)

## 常见问题

**Q: 服务启动失败？**
A: 检查是否正确设置了 API 密钥，确保 3000 端口未被占用

**Q: Xcode 还是连不上？**
A: 确认服务正在运行，Base URL 填写正确：`http://localhost:3000`或端口填写正确：`3000`

**Q: Python 版本需要什么依赖？**
A: 需要 Python 3.8+，依赖已列在 requirements.txt 中

## 致谢：

- [Xcode AI Proxy](https://github.com/fengjinyi98/xcode-ai-proxy)
- [Kimi](https://www.kimi.com/)
- [DeepSeek](https://www.deepseek.com/)
- [ZhipuAI](https://bigmodel.cn/)
