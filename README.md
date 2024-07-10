# Rosemary: A Template Engine for Generative AIs
![image](logo.jpg)
Rosemary is a template engine and prompt executor designed especially for generative AIs including chat completions, image generations and more.

The core concept of Rosemary is to treat the interaction with AIs as some functions. By encapsulating the data handling and AI model execution, you can then build any system you want just like building any other software system.

The ultimate goal of rosemary is to provide a novel way to achieve an AI-based system: efficient, flexible, minimum AI-specific knowledge required, and easy to visualize and debug.

## Who is Rosemary for?
You are a good fit for Rosemary if you:
- Want to build a reactive AI system but have limited knowledge of AI or AI-related libraries.
- Want to quickly prototype a generative AI system.
- Are annoyed by the boilerplate code in your AI project and want to find a way to simplify it.
- Want to introduce your system and prompts to others who are not familiar with AI, and even invite them to participate in the system.

## Key Features
- **Templates** - Rosemary use its unique template language to define the interaction, including the input formatting and output processing in a more declarative way. You can even write 'Template for templates' to save more of your time!
- **Configurable** - The configuration you can use on the AI model is just the same as you can use on the provider's official API.
- **Extensible** - The end-point provider you like has not been supported by Rosemary? You can easily add the new API by yourself, even if it is a local model!
- **Modular** - Manage all your templates in separate files, and use namespaces to keep them organized.
- **Streaming** - Streaming support for building real-time applications.
- **Multi-modal** - Multi-modal support, including text, image, and more.

## Quick Start
### Installation
First, install Rosemary via pip:
```bash
pip install rosemary_ai
```
Rosemary is an under-development project, so you may want to usually check and update the package:
```bash
pip install --upgrade rosemary_ai
```

### Get API Key
The following example uses OpenAI's GPT as the AI model. You need to get an API key from [OpenAI](https://beta.openai.com/signup/). After you get the key, you can set it as an environment variable:
```bash
export OPENAI_API_KEY=your-api-key
```
Or you can directly set the key in your python code, which will be shown later.

### Write Templates
Rosemary has its own template language called Rosemary Markup Language (RML). Create a new file named `hello world.rml` with the following content:
```html
<import path="common"/>

<petal name="hello" var="name">
    <formatter>
        <gpt.chat>
            <message role="'system'">
                Say "Hello, {name}!"
            </message>
        </gpt.chat>
    </formatter>
</petal>
```
This template defines a function ("petal" as we call it in Rosemary) named `hello` that takes a variable `name` and returns a message "Hello, {name}!". Of course, the message will actually be generated by the GPT model.

The `<formatter>` tag tells Rosemary how to use the input variable to create the message sent to the AI model. To send the message to GPT, a specific form of data is required. However, Rosemary provides a set of build-in templates which can save you from writing the data format manually. In this case, we use the `<gpt.chat>` template to format the message.

### Load Templates
Now that we have the template, the next step is to load it in Python. Create a new Python file named `main.py` in the same directory as the RML file, and write the following code:
```python
from rosemary_ai import load

load('test1', 'hello world.rml')
```
This code loads the template file `hello world.rml` to the system and gives it a name `test1`, which can be used to refer to this specific template later.

### Assign Petals to Functions
Now, let's assign the `hello` petal to a function. Add the following code to `main.py`:
```python
from rosemary_ai import petal

@petal('test1', 'hello', model_name='gpt-3.5-t')
def hello(name: str) -> str:
    pass
```

It is a bit weird, isn't it? In a nutshell, the `@petal` decorator makes the 'hello' function in python work just as the 'hello' petal we defined in the RML file ('test1'). The body of the function is thus useless. However, it is helpful to write correct parameters and type hints for further usage.

The `model_name` parameter specifies the AI model to use. In this case, we use OpenAI's GPT-3.5 Turbo model. You can also decide which model to use each time you call the function, but we will keep it simple for now.

### Execute the Function
The code to execute the function is as simple as you can imagine:
```python
print(hello('Alice'))
```

> if you haven't set the `OPENAI_API_KEY` environment variable, you can set it in the code now:
> ```python
> print(hello('Alice', options={'api_key': your_api_key}))
> ```

If you run the `main.py` file, you will probably see the output "Hello, Alice!". Sometimes it may generate a different message, but that's the fun part of generative AI, isn't it?

## Further Information
You can find more syntax, examples and the API reference on the [official wiki](https://github.com/snw2015/Rosemary-AI/wiki)
(under construction).

## Model support
Rosemary currently supports the following AI models (more models will be added soon!):

| Provider | Model         | In-package Model Name |
|----------|---------------|-----------------------|
| OpenAI   | GPT-3.5 Turbo | gpt-3.5-t             |
| OpenAI   | GPT-4 Turbo   | gpt-4-t               |
| OpenAI   | GPT-4o        | gpt-4o                |
| OpenAI   | DALL-E 3      | dall-e-3              |

## Roadmap
Rosemary is still in the early stage of development. We are looking forward to adding more features and improving the usability of the system. Here are some of the features we are planning to add in the future:

- Controllable logging and exception handling
- More AI endpoint support
- AsyncIO support
- Stronger and more flexible template language
- More built-in templates
- Visualize and debug tools

If you have any suggestions or feature requests, feel free to open an issue on GitHub or contact us directly.

## License
Rosemary is licensed under the [MIT License](LICENSE).
