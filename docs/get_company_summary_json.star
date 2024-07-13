```python
company = messages()[-1]
answers = []
search_results = search(company)
company_summary = llm_call([search_results], "summarize what the company does and what products it sells")
answers.append(company_summary)
founder_search = search(f"{company} founders executive team", 3)
for founder_result in founder_search:
    founder = llm_call([founder_result], f"extract the names and positions of founders and executives that work at {company}")
    answers.append(founder)
result = llm_call(answers, 'there is a company summary and then a list of people who are the founders and the executive team. Simplify this into json with the structure {"company": "company name", "summary": "the company summary", "product_descriptions": ["product 1 description", "product 2 description"], "executives": [{"name": "person name 1", "title": "person title"}, {"name": "person name 2", "title": "vp of product"}]}')
answer(result, False)
```
