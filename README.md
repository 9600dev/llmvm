# LLM Stack Based VM

A prototype to demonstrate a natural language -> Abstract Syntax Tree -> Stack based Virtual Machine execution, where ChatGPT/Llama2 is cooperatively running VM instructions and helping plan xecution flow.

## The Problem

ChatGPT supports 'function calling' by passing a query (e.g. "What's the weather in Boston") and a JSON blob with the signatures of supporting functions available to be called locally (i.e. def get_weather(location: str)...). Examples seen [here](https://medium.com/@lucgagan/understanding-chatgpt-functions-and-how-to-use-them-6643a7d3c01a).

However, this interation is usually Task -> LLM decides what helper function to call -> Call helper function -> Work with result. And does not allow for arbitrary decontruction of a task into a series of helper function calls that can be intermixed with both control flow, or cooperative sub-task execution.

This prototype shows that LLM's are capable of taking a user task, reasoning about how to decontruct the task into sub-tasks, understanding how to schedule and execute those sub-tasks on its own or with via a virtual machine, and working with the VM to resolve error cases.

The LLM is able to build a mental-model of the Stack Based Virtual Machine through a natural language definition alone; emit an AST that runs on that VM through an EBNF grammar definition and many-shot examples; and then work with the VM to progress through sub-task execution through User Message -> Assistant Message -> User Message Chat interactions.

## Examples:

Input:
> "I'll give you a list of names and companies. I want you to summarize their careers and contact details: Bill Jia - Meta, Elise McKay - Pendal Group, Jeff Dean - Google."

Transformation to AST:

```
function_call(WebHelpers.search_linkedin_profile("Bill", "Jia", "Meta"))
llm_call("Summarize career profile and contact details")
answer(stack_pop(1))
function_call(WebHelpers.search_linkedin_profile("Erik", "Meijer", "Microsoft"))
llm_call("Summarize career profile and contact details")
answer(stack_pop(1))
function_call(WebHelpers.search_linkedin_profile("Jeff", "Dean", "Google"))
llm_call("Summarize career profile and contact details")
answer(stack_pop(1))
```

Input:

> "Open my contacts, find Leo Huang, Jeff Dean and Micheal Jones and set up a meeting with them for tomorrow at 1pm"

Transformation:

```
function_call(webhelpers.search_linkedin_profile("leo", "huang", ""))
function_call(webhelpers.search_linkedin_profile("jeff", "dean", ""))
function_call(webhelpers.search_linkedin_profile("micheal", "jones", ""))
llm_call(stack_pop(3), "extract the email addresses of leo huang, jeff dean, and micheal jones")
function_call(emailhelpers.send_calendar_invite("your name", "your_email@example.com", ["leo_huang@example.com", "jeff_dean@example.com", "micheal_jones@example.com"], "meeting", "meeting details", "tomorrow at 1pm", "tomorrow at 2pm"))

```

## EBNF Grammar

```
<program> ::= { <statement> }
<statement> ::= <llm_call> | <foreach> | <function_call> | <data> | <answer> | <set> | <get> | <uncertain_or_error>
<llm_call> ::= 'llm_call' '(' [ <stack> ',' <text> | <stack_pop> ',' <text> | <text> ] ')'
<foreach> ::= 'foreach' '(' [ <stack> | <stack_pop> | <data> ] ',' <statement> ')'
<function_call> ::= 'function_call' '(' <helper_function> ')'
<data> ::= 'data' '(' [ <list> | <stack> | <stack_pop> | <text> ] ')'
<answer> ::= 'answer' '(' [ <stack> | <stack_pop> | <text> ] ')'
<get> ::= 'get' '(' <text> ')'
<set> ::= 'set' '(' <text> ')'
<text> ::= '"' { <any_character> } '"'
<list> ::= '[' { <text> ',' } <text> ']'
<stack> ::= 'stack' '(' ')'
<stack_pop> ::= 'stack_pop' '(' <digit> ')'
<digit> ::= '0'..'9'
<any_character> ::= <all_printable_characters_except_double_quotes>
<helper_function> ::= <function_call_from_available_helper_functions>
```

Explanation of nodes:

```llm_call```: A call back to you, the Large Language Model (LLM) that reads the argument, tries to evaluate it, and pushes the response from the LLM on the top of the stack. Emit this node when you feel like you can contribute to the interpretation and execution of the argument supplied.

```foreach```: For each list item defined in the dataframe node, or a list that exists on the stack, or the entire stack as a list, execute the statement and push each of the results of statement execution on to the stack.

```function_call```: A function call you would like the virtual machine to execute for you. The function_call can only specify a function that is listed in "List of Functions:" below. There are no other functions the virtual machine can call. The result of the function call is pushed on top of the stack.

```answer```a direct answer or result from you that does not need further evaluation. answer nodes will be shown to the user, should be relavent to the users problem or question. answer nodes are not pushed on the stack. You can also use an answer node to pop all elements off the stack and into an answer node that will be shown to the user in response to their question or problem. To do this, use answer(stack()). You can also pop the top of the stack into an answer node using answer(stack()).

```dataframe```: use this to specify a list e.g. ["1", "2", "3"] or ["hello", "world"] or use dataframe(stack()) to coherce the all elements on the stack into a dataframe.

```stack```: represents all the elements or data on the execution stack. This node will pop all elements off the stack.

```stack_pop```: this node will pop off 'n' elements from the stack. Typically you only need to pop 1 element off, using stack_pop(1). You can pop more elements off the stack, eg 2 elements: stack_pop(2).

```set```: this node pops one element from the stack and copies the element into the supplied variable name. E.g set("var1") will pop the top of the stack and copy the element to "var1". It does not push anything on the stack. Use this node for storage of results for use at a later part of AST execution.

```get```: this node retrieves the element in the supplied variable name and pushes it on to the stack. i.e. get("var1") will get the node in "var1" and push it on the stack.

## List of available helper functions:

Any function can be a helper function, so long as the argument types are simple types, and the output type is a string.

```
WebHelpers.get_url(url, force_firefox)  # Url can be a http or https web url or a filename and directory location.
WebHelpers.get_news_article(url)  # Extracts the text from a news article
WebHelpers.get_url_firefox(url)  # This is useful for hard to extract text, an exception thrown by the other functions,
or when searching/extracting from sites that require logins liked LinkedIn, Facebook, Gmail etc.
WebHelpers.search_news(query, total_links_to_return)  # Searches the current and historical news for a query and returns the text of the top results
WebHelpers.search_internet(query, total_links_to_return)  # Searches the internet for a query and returns the text of the top results
WebHelpers.search_linkedin_profile(first_name, last_name, company_name)  # Searches for the LinkedIn profile of a given person name and optional company name and returns the profile text
WebHelpers.get_linkedin_profile(linkedin_url)  # Extracts the career information from a person's LinkedIn profile from a given LinkedIn url
EdgarHelpers.get_latest_form_text(symbol, form_type)  # This is useful to get the latest financial information for a company,
their current strategy, investments and risks.
PdfHelpers.parse_pdf(url_or_file)  # You can only use either a url or a path to a pdf file.
MarketHelpers.get_stock_price(symbol)  # Get the current or latest price of the specified stock symbol
MarketHelpers.get_market_capitalization(symbol)  # Get the current market capitalization of the specified stock symbol
EmailHelpers.send_email(sender_email, receiver_email, subject, body)  # Send an email from sender to receiver with the specified subject and body text
EmailHelpers.send_calendar_invite(from_name, from_email, attendee_emails, subject, body, start_date, end_date)  # Send a calendar invite to the attendee
```

## How the interpreter handles input larger than context window:

Given:

```
function_call(WebHelpers.get_url("https://attract.ai/about-us/"))
llm_call(stack_pop(1), "From the content provided, Extract a list of all people names and the company name they work at")
```

where the HTML returned from https://attract.ai/about-us/ is larger than the context window, the interpreter will:

* Split the content into 512 token chunks based on sentence chunking.
* Randomly select 20% of the chunks, and ask the LLM to evaluate if the **all** chunks contain data relevant to the query task, in this case "extract a list of all people names ...".
* If true:
  * Run the LLM query in map reduce style, mapping all chunks to query, and reducing through summation. (slow and expensive)
* If false:
  * Chunk and perform FAISS based similarity, select top(n) chunks and perform query. (fast and cheap)

## Extra thoughts:

* Programs (or AST's) are arbitrarily composable. Executed programs can be serialized and later deserialized and pushed on the runtime stack as input to other programs.
* Error handling isn't implemented yet. Basic approach: if error occurs in AST node interpretation, either:
  * retry
  * propagate error up and allow parent node to handle
  * add extra context to interpretation (current stack is used for most context, but previous results from prior execution could be used.)
  * ask LLM to re-write AST from parent to try and improve probability of successful execution.

# Install

* Install pyenv: ```curl https://pyenv.run | bash```
* ```pyenv install 3.11.4```
* ```pyenv virtualenv 3.11.4 llmasm```
* Install poetry: ```curl -sSL https://install.python-poetry.org | python3 -```
* ```poetry config virtualenvs.prefer-active-python true``
* ```poetry install```
* python repl.py