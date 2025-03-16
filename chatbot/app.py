from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    endpoint_url="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)




examples = [


  { 

    "instruction": "What is Bignalytics?", 

    "input": "Tell me about Bignalytics", 

    "output": "Bignalytics IT Solutions is one of India’s fastest-growing organizations, providing job-oriented training programs in cutting-edge technologies. Their mission is to empower students and professionals with the skills needed to excel in the real world. They offer top-notch training in master’s data science, advanced data analytics, machine learning, AI, and GenAI. The curriculum is designed around live industry projects to solve real-world challenges. Many learners have secured jobs in leading IT companies like Capgemini, TCS, Deloitte, and Accenture." 

  }, 

  { 

    "instruction": "List the courses offered by Bignalytics", 

    "input": "What courses does Bignalytics offer?", 

    "output": "Bignalytics offers the following courses:\n1. Master’s Program in Machine Learning and Artificial Intelligence\n2. Master’s Program in Data Science and Machine Learning\n3. Certificate Program in Python Programming\n4. Certified Professional in Ethical Hacking" 

  }, 

  { 

    "instruction": "Describe the Master’s Program in Machine Learning and Artificial Intelligence", 

    "input": "What is covered in the Master’s Program in Machine Learning and AI?", 

    "output": "The Master’s Program in Machine Learning and Artificial Intelligence covers advanced ML and AI techniques, including supervised & unsupervised learning, deep learning, NLP, computer vision, and AI model deployment." 

  }, 

  { 

    "instruction": "Describe the Master’s Program in Data Science and Machine Learning", 

    "input": "What will I learn in the Data Science and Machine Learning Master’s Program?", 

    "output": "The Master’s Program in Data Science and Machine Learning provides training in Python & SQL for data science, data cleaning, feature engineering, predictive modeling, and data visualization with Power BI and Tableau." 

  }, 

  { 

    "instruction": "Describe the Certificate Program in Python Programming", 

    "input": "Tell me about the Python Programming Certificate Program", 

    "output": "The Certificate Program in Python Programming is designed for beginners and covers Python basics, data structures, OOP, file handling, automation, and data analysis with Pandas & NumPy." 

  }, 

  { 

    "instruction": "Describe the Certified Professional in Ethical Hacking program", 

    "input": "What topics are covered in the Ethical Hacking certification?", 

    "output": "The Certified Professional in Ethical Hacking program focuses on cybersecurity and ethical hacking concepts such as network security, penetration testing, cryptography, malware analysis, and web application security." 

  }, 

  { 

    "instruction": "How can I enroll in a course at Bignalytics?", 

    "input": "How do I register for a course at Bignalytics?", 

    "output": "To enroll in a course, visit the official Bignalytics website, choose your desired program, and complete the registration process. You can also contact their support team for assistance." 

  }, 

  { 

    "instruction": "Does Bignalytics provide certificates after course completion?", 

    "input": "Will I get a certificate after completing a course?", 

    "output": "Yes, Bignalytics provides industry-recognized certificates upon successful course completion, which can help enhance your resume and job prospects." 

  }, 

  { 

    "instruction": "Do Bignalytics courses include real-world projects?", 

    "input": "Will I work on live projects during my course?", 

    "output": "Yes, all Bignalytics courses include hands-on, industry-relevant projects that help learners apply theoretical knowledge to real-world scenarios." 

  }, 

  { 

    "instruction": "What job opportunities are available after completing a course at Bignalytics?", 

    "input": "Can I get a job after completing a Bignalytics course?", 

    "output": "Many Bignalytics graduates have secured positions in top IT companies like Capgemini, TCS, Ksolves, Yash Technologies, Deloitte, and Accenture. The courses are designed to make you job-ready in the field of data science and AI." 

  }, 

  { 

    "instruction": "What is Bignalytics’ refund policy?", 

    "input": "Can I get a refund after enrolling in a course?", 

    "output": "As per Bignalytics’ company policy, once a student has completed admission/enrollment, there is no provision for a return, exchange, or refund of the course. The course will be available only to the enrolled student. If any payment issues occur during online transactions, students must register a ticket by contacting Bignalytics at contact@bignalytics.in or calling the numbers provided on their website. The support team will respond within 48 working hours." 

  }, 

  { 

    "instruction": "What are the payment options for Bignalytics courses?", 

    "input": "How can I pay for a course at Bignalytics?", 

    "output": "Bignalytics accepts both online and cash payments. Candidates must pay for the course as per the plan instructed by the admission team." 

  } 

]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}\n"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a helpful assistant for Bignalytics. Answer the following questions:",
    suffix="Q: {question}\nA:",
    input_variables=["question"]
)

st.title("BIGNALYTICS - AI Assistant")

input_text = st.text_input("Ask a question about Bignalytics:")

output_parser = StrOutputParser()


chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)