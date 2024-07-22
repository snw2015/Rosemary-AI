# chat completion models
# register_generator(['gpt-3.5-t', 'gpt-3.5-turbo'], GPTChatGenerator('gpt-3.5-turbo'))
# register_generator('gpt-4', GPTChatGenerator('gpt-4'))
# register_generator(['gpt-4-t', 'gpt-4-turbo'], GPTChatGenerator('gpt-4-turbo'))
# register_generator('gpt-4o', GPTChatGenerator('gpt-4o'))
# register_generator(['claude-3.5-s', 'claude-3.5-sonnet'], ClaudeChatGenerator('claude-3-5-sonnet-20240620'))
# register_generator(['claude-3-h', 'claude-3-haiku'], ClaudeChatGenerator('claude-3-haiku-20240307'))
# register_generator(['claude-3-s', 'claude-3-sonnet'], ClaudeChatGenerator('claude-3-sonnet-20240229'))
# register_generator(['claude-3-o', 'claude-3-opus'], ClaudeChatGenerator('claude-3-opus-20240229'))

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
