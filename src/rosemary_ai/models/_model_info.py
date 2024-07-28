GPT = {
    'gpt-3.5-turbo': ['gpt-3.5-turbo', 'gpt-3.5-t'],
    'gpt-4': ['gpt-4'],
    'gpt-4-turbo': ['gpt-4-turbo', 'gpt-4-t'],
    'gpt-4o': ['gpt-4o'],
    'gpt-4o-mini': ['gpt-4o-mini', 'gpt-4o-m']
}

CLAUDE = {
    'claude-3-5-sonnet-20240620': ['claude-3.5-s', 'claude-3.5-sonnet'],
    'claude-3-haiku-20240307': ['claude-3-h', 'claude-3-haiku'],
    'claude-3-sonnet-20240229': ['claude-3-s', 'claude-3-sonnet'],
    'claude-3-opus-20240229': ['claude-3-o', 'claude-3-opus']
}

COMMAND = {
    'command-r-plus': ['command-r-p', 'command-r-plus'],
    'command-r': ['command-r'],
    'command': ['command'],
    'command-nightly': ['command-n', 'command-nightly'],
    'command-light': ['command-l', 'command-light'],
    'command-light-nightly': ['command-ln', 'command-light-nightly'],
}

# image generation models
# register_generator('dall-e-3', GPTImageGenerator('dall-e-3'))

DALL_E = {
    'dall-e-3': ['dall-e-3'],
    'dall-e-2': ['dall-e-2']
}

STABLE_GEN_V2 = {
    'stable-diffusion-ultra': ['sd-ultra', 'sd-u'],
    'stable-diffusion-core': ['sd-core', 'sd-c'],
    'stable-diffusion-3-large': ['sd3-large', 'sd3-l'],
    'stable-diffusion-3-large-turbo': ['sd3-large-turbo', 'sd3-l-t'],
    'stable-diffusion-3-medium': ['sd3-medium', 'sd3-m'],
}

STABLE_GEN_V1 = {
    'stable-diffusion-xl': ['sd-xl', 'sdxl'],
    'stable-diffusion-1.6': ['sd-1.6', 'sd'],
    'stable-diffusion-beta': ['sd-beta'],
}

# Embedding models

OPENAI_EMBEDDINGS = {
    'text-embedding-3-small': ['gpt-embed-3s'],
    'text-embedding-3-large': ['gpt-embed-3l'],
    'text-embedding-ada-002': ['gpt-embed-ada'],
}
