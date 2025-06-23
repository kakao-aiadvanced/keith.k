from openai import OpenAI

client = OpenAI()

def completion(content):
	completion = client.chat.completions.create(
		model="gpt-4o-mini",
		messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{
				"role": "user",
				"content": content.strip()
			}
		]
	)

	print(completion.choices[0].message.content)

def getting_started():
	completion("What is the capital of France?")

def prompt():
	completion("What are the best places in Korea?")
	completion("What are the tourist attractions in Korea?")

def few_shot():
	completion("""
		Translate English to Korean.
		cat => 고양이, fox => 여우, horse => 말, wolf => 늑대, bear => 곰
		dog
	""")

def few_shot2():
	completion("""
		다음 문장이 긍정인지 부정인지 판단해줘.
		1. This movie is pretty good. : positive
		2. The actor was terrible. : negative
		3. The plot was boring and predictable. : negative
		4. This is amazing. : positive
		5. I was sleeping. : negative
		The storyline was dull and uninspiring.
	""")

def few_shot3():
	completion("""
		Convert the following natural language requests into SQL queries:
		1. "Find employees having a salary over 50000": SELECT * FROM employees WHERE salary > 50000;
		2. "재고가 없는 상품을 찾아줘": SELECT * FROM products WHERE stock = 0;
		3. "수학 점수가 90 넘는 학생을 찾아줘": SELECT name FROM students WHERE math_score > 90;
		4. "Find recent orders within 30 days": SELECT * FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);
		5. "도시별 고객 수를 계산해줘": SELECT city, COUNT(*) FROM customers GROUP BY city;

		Request: "Find the average salary of employees in the marketing department."
		SQL Query:
	""")

def cot1():
	completion("""
		# Simple - 1
		Solve the following problem step-by-step: 23 + 47

		Step-by-step solution:
		1. Write your Prompt
		2. Write your Prompt
		3. Write your Prompt
		4. Write your Prompt

		Answer: 70
	""")


# getting_started()
# prompt()
# few_shot()
# few_shot2()
# few_shot3()
cot1()
