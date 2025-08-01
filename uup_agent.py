import json
import re
from typing import Dict, Any, Optional
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter
from pymongo import MongoClient
from uup_config import Config


class MongoAgent:
    def __init__(self):
        self.config = Config()
        
        self.source_client = MongoClient(self.config.SOURCE_MONGODB_URI)
        self.source_db = self.source_client[self.config.SOURCE_DB_NAME]
        self.source_collection = self.source_db[self.config.SOURCE_COLLECTION_NAME]
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.MODEL_NAME,
            google_api_key=self.config.GOOGLE_API_KEY,
            temperature=0.001,
        )
        self._setup_prompts()
    
    def _setup_prompts(self):
        self.generate_query_template = '''You are a AI agent which is proficient with the MongoDB database.
Given an input question, You have to understand the user query and then according to the query if it is question related to his transactions you have to create a syntactically correct MongoDB query to run., else respond as helpful transaction agent.
limit your query to at most 15 results for the query related to person, if the user specifies the numbers use that.
You can sort the results by a relevant field to return the most interesting examples in the database.
Never query for all documents without filters, always ask for relevant fields given the question.

For the collection {collection_name}, the fields and their descriptions are: {fields_description} \n ---- \n
You have to focus in the fields which are given below with their description: {collection_info}
        
DO NOT make any write operations (insertOne, updateOne, deleteOne, drop, etc.) to the update or modify the database.
If user asks to modify the database, respond with: "This is out of my capabilities. You can ask questions about the transaction history."
        
Always return MongoDB queries in the following JSON format:
{{
"operation": "find|aggregate|count",
"filter": {{}},
"projection": {{}},
"sort": {{}},
"limit": 15,
"pipeline": []
}}
        
If user is asking general question you have to answer them in simple manner.
And asking about the query they want to asking about their transaction database. 
Below are examples you can refer to the example to query the question:
{examples}
        
Question: {question}
Query:'''
        
        self.generate_query_prompt = PromptTemplate.from_template(self.generate_query_template)
        
        self.answer_prompt = PromptTemplate(
            input_variables=["question","result"],
            template="""You are a helpful assistant that will answer the user question using the Query Result. Query result the result after running the mongodb query.  

Original Question: {question} \n ---- \n
Query Result: {result}
            
Please provide a understandable and simple response to the user. And add all the valid results in answer. Answer in simple sentence.
If the result is empty, explain that no matching records were found.
If there's an error, explain what went wrong in simple terms.
If user has question about the month and if in query result if you get the month referring in number you have to convert them back into their name like January, February and other.
The transaction amount is in Indian Rupees (INR) currency.

Response:"""
        )

        self.chat_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a helpful assistant that will assist the user in querying about their transaction documents uploaded by them. You have to answer their question in a polite and helpful manner.

Original Question: {question}

Response:"""
        )

        examples = [
  {
 'input': "How much money have I debited in total?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":null, "total_debited":{"$sum":"$Amount_debited_num"} } } ] }\n----''' 
},
{
 'input': "Show me all my credit transactions.",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Amount_credited":{"$ne":"0"}}, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "What payments have I made in the 'Shopping' category?",
'Mongodb_Query': '''{ "operation":"find", "filter":{ "Categories":"Shopping", "Amount_debited":{"$ne":"0"} }, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "What are my top 5 expenses by merchant?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{"Amount_debited":{"$ne":"0"}} }, { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":"$Merchant", "total_spent":{"$sum":"$Amount_debited_num"} } }, { "$sort":{"total_spent":-1} }, { "$limit":5 } ] }\n----''' 
},
{
 'input': "How many transactions did I have with 'RazorpaySo'?",
'Mongodb_Query': '''{ "operation":"count", "filter":{"Merchant":"RazorpaySo"}, "projection":{}, "sort":{}, "limit":0, "pipeline":[] }\n----''' 
},
{
 'input': "List all transactions made via 'NEFT' in January 2025.",
'Mongodb_Query': '''{ "operation":"find", "filter":{ "Mode_of_Payment":"NEFT", "Date":{"$regex":"^\\d{2}/01/2025$"} }, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "Show me all income transactions for 'Compensation_Salaries'.",
'Mongodb_Query': '''{ "operation":"find", "filter":{ "Categories":"Compensation_Salaries", "Amount_credited":{"$ne":"0"} }, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "What was my highest single debit transaction?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{"Amount_debited":{"$ne":"0"}} }, { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$sort":{"Amount_debited_num":-1} }, { "$limit":1 } ] }\n----''' 
},
{
 'input': "Show me all transactions on '05/15/2025'.",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Date":"05/15/2025"}, "projection":{}, "sort":{}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "How many transactions are categorized as 'Travel'?",
'Mongodb_Query': '''{ "operation":"count", "filter":{"Categories":"Travel"}, "projection":{}, "sort":{}, "limit":0, "pipeline":[] }\n----''' 
},
{
 'input': "What is the total amount credited via 'IMPS'?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{ "Mode_of_Payment":"IMPS", "Amount_credited":{"$ne":"0"} } }, { "$addFields":{ "Amount_credited_num":{"$toDouble":"$Amount_credited"} } }, { "$group":{ "_id":null, "total_credited":{"$sum":"$Amount_credited_num"} } } ] }\n----''' 
},
{
 'input': "List all transactions with 'Paytm'.",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Merchant":"Paytm"}, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "Show me transactions between $1000 and $5000 that were debits.",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$match":{ "Amount_debited_num":{"$gte":1000,"$lte":5000} } }, { "$sort":{"Date":-1} }, { "$limit":15 } ] }\n----''' 
},
{
 'input': "Which categories have I spent on recently? (last 10 debit transactions)",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Amount_debited":{"$ne":"0"}}, "projection":{"Categories":1,"Date":1,"Amount_debited":1}, "sort":{"Date":-1}, "limit":10, "pipeline":[] }\n----''' 
},
{
 'input': "Give me the total credited amount for each category.",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{"Amount_credited":{"$ne":"0"}} }, { "$addFields":{ "Amount_credited_num":{"$toDouble":"$Amount_credited"} } }, { "$group":{ "_id":"$Categories", "total_credited":{"$sum":"$Amount_credited_num"} } }, { "$sort":{"total_credited":-1} } ] }\n----''' 
},
{
 'input': "Show me transactions from 'IndiaIdeas' that are not 'Debit'.",
'Mongodb_Query': '''{ "operation":"find", "filter":{ "Merchant":"IndiaIdeas", "Amount_credited":{"$ne":"0"} }, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "How many transactions occurred in the 'Others' category?",
'Mongodb_Query': '''{ "operation":"count", "filter":{"Categories":"Others"}, "projection":{}, "sort":{}, "limit":0, "pipeline":[] }\n----''' 
},
{
 'input': "What was the total amount spent on 'Material_and_Supplies' in 2025?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{ "Categories":"Material_and_Supplies", "Amount_debited":{"$ne":"0"}, "Date":{"$regex":"/2025$"} } }, { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":null, "total_spent":{"$sum":"$Amount_debited_num"} } } ] }\n----''' 
},
{
 'input': "List my 10 most recent transactions, showing merchant and amount.",
'Mongodb_Query': '''{ "operation":"find", "filter":{}, "projection":{ "Merchant":1, "Amount_credited":1, "Amount_debited":1, "Date":1 }, "sort":{"Date":-1}, "limit":10, "pipeline":[] }\n----''' 
},
{
 'input': "Which merchants have I received money from?",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Amount_credited":{"$ne":"0"}}, "projection":{"Merchant":1,"Amount_credited":1,"Date":1}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "Show me all transactions with an amount debited greater than $10000.",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$match":{ "Amount_debited_num":{"$gt":10000} } }, { "$sort":{"Amount_debited_num":-1} }, { "$limit":15 } ] }\n----''' 
},
{
 'input': "What is the average debit amount for 'Travel' category?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{ "Categories":"Travel", "Amount_debited":{"$ne":"0"} } }, { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":null, "average_debit":{"$avg":"$Amount_debited_num"} } } ] }\n----''' 
},
{
 'input': "Count how many transactions were made using 'RazorpaySo'.",
'Mongodb_Query': '''{ "operation":"count", "filter":{"Merchant":"RazorpaySo"}, "projection":{}, "sort":{}, "limit":0, "pipeline":[] }\n----''' 
},
{
 'input': "List transactions from the last 7 days.",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$addFields":{ "dateObj":{ "$dateFromString":{ "dateString":{ "$concat":[ {"$substr":["$Date",6,4]},"-", {"$substr":["$Date",3,2]},"-", {"$substr":["$Date",0,2]} ] } } } } }, { "$match":{ "dateObj":{ "$gte":{"$dateSubtract":{"startDate":"$$NOW","unit":"day","amount":7}} } } }, { "$sort":{"dateObj":-1} }, { "$limit":15 } ] }\n----''' 
},
{
 'input': "Show me all 'Other_Expenses' from February 2025.",
'Mongodb_Query': '''{ "operation":"find", "filter":{ "Categories":"Other_Expenses", "Date":{"$regex":"^\\d{2}/02/2025$"} }, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "Show me all transactions above ₹1000",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$addFields":{ "Amount_credited_num":{"$toDouble":"$Amount_credited"}, "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$match":{ "$or":[ {"Amount_debited_num":{"$gt":1000}}, {"Amount_credited_num":{"$gt":1000}} ] } }, { "$sort":{"Date":-1} }, { "$limit":15 } ] }\n----''' 
},
{
 'input': "Show me top 3 merchants by total transaction amount",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$addFields":{ "Amount_credited_num":{"$toDouble":"$Amount_credited"}, "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":"$Merchant", "total_amount":{"$sum":{"$add":["$Amount_credited_num","$Amount_debited_num"]}} } }, { "$sort":{"total_amount":-1} }, { "$limit":3 } ] }\n----''' 
},
{
 'input': "Count transactions by payment mode",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$group":{ "_id":"$Mode_of_Payment", "transaction_count":{"$sum":1} } }, { "$sort":{"transaction_count":-1} } ] }\n----''' 
},
{
 'input': "Show me daily spending totals for debit transactions",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{"Amount_debited":{"$ne":"0"}} }, { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":"$Date", "daily_spending":{"$sum":"$Amount_debited_num"} } }, { "$sort":{"_id":-1} } ] }\n----''' 
},
{
 'input': "Find all income transactions",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Amount_credited":{"$ne":"0"}}, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "What is the average transaction amount by category?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$addFields":{ "Amount_credited_num":{"$toDouble":"$Amount_credited"}, "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":"$Categories", "average_amount":{"$avg":{"$add":["$Amount_credited_num","$Amount_debited_num"]}} } }, { "$sort":{"average_amount":-1} } ] }\n----''' 
},
{
 'input': "Show me transactions from 20/01/2025",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Date":"20/01/2025"}, "projection":{}, "sort":{}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "Calculate total credited vs debited amounts",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$addFields":{ "Amount_credited_num":{"$toDouble":"$Amount_credited"}, "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":null, "total_credited":{"$sum":"$Amount_credited_num"}, "total_debited":{"$sum":"$Amount_debited_num"} } }, { "$project":{ "total_credited":1, "total_debited":1, "net_balance":{"$subtract":["$total_credited","$total_debited"]} } } ] }\n----''' 
},
{
 'input': "Find UPI transactions",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Mode_of_Payment":"UPI"}, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "What is the total number of transactions",
'Mongodb_Query': '''{ "operation":"count", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[] }\n----''' 
},
{
 'input': "Show me IFT transactions",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Mode_of_Payment":"IFT"}, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "What is the total number/count of debit and credit transactions",
'Mongodb_Query': '''{
  "operation": "aggregate", "filter": {},  "projection": {},  "sort": {},  "limit": 0,  "pipeline": [    {"$group": {"_id": null,"debit_count": {"$sum": {"$cond": [{ "$ne": ["$Amount_debited", "0"] },1,0]}},
        "credit_count": {
          "$sum": {
            "$cond": [
              { "$ne": ["$Amount_credited", "0"] },
              1,
              0
            ]
          }
        }
      }
    }
  ]
}\n----''' 
},
{
 'input': "What are the modes of payment I have used?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$group":{ "_id":"$Mode_of_Payment" } }, { "$sort":{"_id":1} } ] }\n----''' 
},
{
 'input': "What types of Categories for which payment is done?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$group":{ "_id":"$Categories" } }, { "$sort":{"_id":1} } ] }\n----''' 
},
{
 'input': "Hello",
'Mongodb_Query': '''Hello! I'm ready to help you with your financial queries. How can I assist you today? you can ask question like: What is the total count of debit and credit transactions? What are the modes of payment I have used? Merchants with whom I have done transactions? What types of Categories for which payment is done?\n----''' 
},
{
 'input': "Show me all transactions for 02/01/2025",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Date":"02/01/2025"}, "projection":{}, "sort":{}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "What's my total shopping expense in January 2025?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{ "Categories":"Shopping", "Amount_debited":{"$ne":"0"}, "Date":{"$regex":"^\\d{2}/01/2025$"} } }, { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":null, "total_spent":{"$sum":"$Amount_debited_num"} } } ] }\n----''' 
},
{
 'input': "Show me all money going out of my account",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Amount_debited":{"$ne":"0"}}, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "Who did I pay the most money to?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{"Amount_debited":{"$ne":"0"}} }, { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":"$Merchant", "total_spent":{"$sum":"$Amount_debited_num"}, "transaction_count":{"$sum":1} } }, { "$sort":{"total_spent":-1} }, { "$limit":1 } ] }\n----''' 
},
{
 'input': "Show me all UPI payments",
'Mongodb_Query': '''{ "operation":"find", "filter":{"Mode_of_Payment":"UPI"}, "projection":{}, "sort":{"Date":-1}, "limit":15, "pipeline":[] }\n----''' 
},
{
 'input': "What is the total sum of credit and debit transactions",
'Mongodb_Query': '''{ "operation":"aggregate", "pipeline":[ { "$group":{ "_id":null, "credit_transaction_count":{ "$sum":{ "$cond":[ {"$gt":[{"$toDouble":"$Amount_credited"},0]}, 1, 0 ] } }, "debit_transaction_count":{ "$sum":{ "$cond":[ {"$gt":[{"$toDouble":"$Amount_debited"},0]}, 1, 0 ] } } } } ] }\n----''' 
},
{
 'input': "What's my total income/credits?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{"Amount_credited":{"$ne":"0"}} }, { "$addFields":{ "Amount_credited_num":{"$toDouble":"$Amount_credited"} } }, { "$group":{ "_id":null, "total_credited":{"$sum":"$Amount_credited_num"} } } ] }\n----''' 
},
{
 'input': "Show me my biggest expenses",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{"Amount_debited":{"$ne":"0"}} }, { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$sort":{"Amount_debited_num":-1} }, { "$limit":10 } ] }\n----''' 
},
{
 'input': "How many times did I pay AvenuesInd?",
'Mongodb_Query': '''{ "operation":"count", "filter":{"Merchant":"AvenuesInd"}, "projection":{}, "sort":{}, "limit":0, "pipeline":[] }\n----''' 
},
{
'input': "Break down my expenses by category",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{"Amount_debited":{"$ne":"0"}} }, { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":"$Categories", "total_spent":{"$sum":"$Amount_debited_num"}, "transaction_count":{"$sum":1} } }, { "$sort":{"total_spent":-1} } ] }\n----''' 
},
{
'input': "Show me all e-commerce purchases over 500 rupees",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$match":{ "Mode_of_Payment":"Ecom" } }, { "$addFields":{ "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$match":{ "Amount_debited_num":{"$gt":500} } }, { "$sort":{"Amount_debited_num":-1} }, { "$limit":15 } ] }\n----''' 
},
{
 'input': "What's the difference between money in and money out?",
'Mongodb_Query': '''{ "operation":"aggregate", "filter":{}, "projection":{}, "sort":{}, "limit":0, "pipeline":[ { "$addFields":{ "Amount_credited_num":{"$toDouble":"$Amount_credited"}, "Amount_debited_num":{"$toDouble":"$Amount_debited"} } }, { "$group":{ "_id":null, "total_credited":{"$sum":"$Amount_credited_num"}, "total_debited":{"$sum":"$Amount_debited_num"} } }, { "$project":{ "net_change":{"$subtract":["$total_credited","$total_debited"]}, "total_credited":1, "total_debited":1 } } ] }\n----''' 
},
]
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{Mongodb_Query}"),
        ])
        
        embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)
        vectorstore = Chroma()
        
        try:
            vectorstore.delete_collection()
        except:
            pass
        
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embeddings,
            vectorstore,
            k=2,
            input_keys=["input"],
        )
        
        self.few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=example_selector,
            input_variables=["input"],
        )
        
        self.rephrase_answer = self.answer_prompt | self.llm | StrOutputParser()
    
    def query_parser(self, query: str) -> Dict:
        try:
            cleaned_query = query.strip()
            cleaned_query = cleaned_query.removeprefix("```json").removesuffix("```")
            cleaned_query = cleaned_query.removeprefix("```").removesuffix("```")
            cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
            
            return json.loads(cleaned_query)
        except json.JSONDecodeError:
            return {"operation": "chat", "response": cleaned_query}
    
    def validate_mongo_query(self, query_dict: Dict) -> int:
        try:
            if not isinstance(query_dict, dict):
                return 2
            
            operation = query_dict.get('operation', '')
            
            if operation == 'chat':
                return 2
            
            if operation not in ['find', 'aggregate', 'count']:
                return 2
            
            if operation == 'aggregate' and 'pipeline' in query_dict:
                pipeline = query_dict['pipeline']
                for stage in pipeline:
                    if isinstance(stage, dict):
                        for key in stage.keys():
                            if key.startswith('$out') or key.startswith('$merge'):
                                return 0
            
            return 1
            
        except Exception:
            return 2
    
    def add_user_filter(self, query_dict: Dict, user_item: str) -> Dict:
        user_filter = {"user.item": user_item}
        
        if query_dict.get('operation') == 'find':
            existing_filter = query_dict.get('filter', {})
            query_dict['filter'] = {**user_filter, **existing_filter}
            
        elif query_dict.get('operation') == 'count':
            existing_filter = query_dict.get('filter', {})
            query_dict['filter'] = {**user_filter, **existing_filter}
            
        elif query_dict.get('operation') == 'aggregate':
            pipeline = query_dict.get('pipeline', [])
            match_stage = {"$match": user_filter}
            
            if pipeline and isinstance(pipeline[0], dict) and '$match' in pipeline[0]:
                existing_match = pipeline[0]['$match']
                pipeline[0]['$match'] = {**user_filter, **existing_match}
            else:
                pipeline.insert(0, match_stage)
            
            query_dict['pipeline'] = pipeline
        
        return query_dict
    
    def get_collection_details(self) -> str:
        field_details = {
            'Date': "(String, format 'DD/MM/YYYY') The date of the transaction.",
            'Mode_of_Payment': '''(String) The method used for the payment (e.g., "NEFT", "IMPS", "UPI")''',
            'Merchant': '''(String) The entity or person involved in the transaction (e.g., "AvenuesInd", "RazorpaySo", "Paytm", "NEHA ROY").''',
            'Categories': '''(String) The classification of the transaction (e.g., "Travel", "Transfers", "Income", "Shopping", "Other_Expenses", "Compensation_Salaries").''',
            'Amount_credited': '''(String) The amount of money added to the account. Amount_credited will be 0 for the records where Amount_debited is present.''',
            'Amount_debited': '''(String) The amount of money deducted from the account. Amount_debited will be 0 for the records where Amount_credited is present.''',
            'user': '''(Object) Contains user information whose transaction data is being processed, including item field for user identification.''',
            'feat': '''(Object) Contains additional features like sort and type information.'''
        }
        
        details = ""
        for field, description in field_details.items():
            details += f"Field Name: {field}\nField Description: {description}\n----\n"
        
        return details
    

    def get_table_info(self, user_item: str):
        try:
            sample_doc = self.source_collection.find_one({"user.item": user_item})
            if sample_doc:
                if '_id' in sample_doc:
                    del sample_doc['_id']
                
                fields = list(sample_doc.keys())
                return {
                    'collection_name': self.config.SOURCE_COLLECTION_NAME,
                    'fields': fields,
                    'sample_document': sample_doc
                }
            return None
        except Exception as e:
            print(f"Error getting table info: {e}")
            return None
    
    def execute_query(self, query_dict: Dict):
        try:
            operation = query_dict.get('operation', 'find')
            filter_criteria = query_dict.get('filter', {})
            projection = query_dict.get('projection', {})
            sort = query_dict.get('sort', {})
            limit = query_dict.get('limit', 0)
            
            
            if operation == 'find':
                cursor = self.source_collection.find(filter_criteria, projection)
                if sort:
                    cursor = cursor.sort(list(sort.items()))
                if limit > 0:
                    cursor = cursor.limit(limit)
                
                results = list(cursor)
                for result in results:
                    if '_id' in result:
                        result['_id'] = str(result['_id'])
                
                return results
                
            elif operation == 'aggregate':
                pipeline = query_dict.get('pipeline', [])
                results = list(self.source_collection.aggregate(pipeline))
                for result in results:
                    if '_id' in result:
                        result['_id'] = str(result['_id'])
                        
                return results
                
            elif operation == 'count':
                count = self.source_collection.count_documents(filter_criteria)
                return count
                
            else:
                return {"error": "Unsupported operation"}
                
        except Exception as e:
            print(f"Error executing query: {e}")
            return {"error": str(e)}
    
    def get_user_info(self, user_item: str) -> Optional[Dict]:
        try:
            sample_record = self.source_collection.find_one({"user.item": user_item})
            if sample_record and "user" in sample_record:
                return sample_record["user"]
            return None
        except Exception as e:
            print(f"Error getting user info: {e}")
            return None

    def process_output(self, output: str) -> str:
        output = output.replace('\n', ' ')
        output = output.replace('*', ' ')

        return output
    
    def process_query(self, question: str, user_item: str) -> str:
        try:
            collection_info = self.get_table_info(user_item)
            collection_name = self.config.SOURCE_COLLECTION_NAME

            # print("collection_info: ", collection_info, '\n')
            # print("collection_name: ", collection_name, '\n')
            # print("self.get_collection_details(): ", self.get_collection_details(), '\n')

            # print("---"*20)
            final_chain = (
                {
                    'examples': RunnableLambda(lambda x: self.few_shot_prompt.format(input=x["question"])),
                    'question': itemgetter("question"), 
                    'collection_name': RunnableLambda(lambda x: collection_name), 
                    'fields_description': RunnableLambda(lambda x: self.get_collection_details()),
                    "collection_info": RunnableLambda(lambda x: collection_info),
                    "user_item": itemgetter("user_item")
                }
                | self.generate_query_prompt
                | self.llm
                | StrOutputParser()
                | RunnableLambda(self.query_parser)
            )
            
            generated_query = final_chain.invoke({'question': question, 'user_item': user_item})
            # print("Generated prompt before user filter:", generated_query)
            
            validation_result = self.validate_mongo_query(generated_query)
            
            if validation_result == 2:
                if isinstance(generated_query, dict) and 'response' in generated_query:
                    return generated_query['response']
                else:
                    chat_response = self.chat_prompt | self.llm | StrOutputParser()
                    return chat_response.invoke({'question': question})
            
            elif validation_result == 0:
                return "I don't have access to modify the data. You can ask other questions about your transactions."
            

            query_with_filter = self.add_user_filter(generated_query, user_item)
            # print("Query with user filter:", query_with_filter)
            
            # Execute the MongoDB query
            mongo_response = self.execute_query(query_with_filter)
            # print("Mongo response:", mongo_response)
            
            response = self.rephrase_answer.invoke({
                'question': question,
                'fields_description': self.get_collection_details(),
                'result': mongo_response
            })
            
            # print('response: ', response)
            response = self.process_output(response)

            return response
            
        except Exception as e:
            print(f"Error in process_query: {e}")
            return f"An error occurred while processing your query: {str(e)}"
    
    def close_connection(self):
        self.source_client.close()

