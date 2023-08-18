#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import bokeh
import numpy

import numpy as np

from sep_core import sep




# In[2]:

general = """Do not assume any more information about yourself. Do not ask for more than one property at a time. Be nice, polite and helpful, translating computer technical terms into everyday language. If you finish, return my answer in the JSON format specified above. Do not generate anything else but the JSON when finished. Delete units, information in brackets and other unnecessary information. Make sure JSON is code readable and have all necessary values as the format specified."""

user_program = """
## What is the Seismic Earth Pressure (SEP) Calculator

An interactive web application has been developed to compute seismic earth pressure. The application employs an expanded form of Rankine's classic earth pressure solution. The application computes seismic active earth pressure behind rigid walls supporting c–φ backfill considering both wall inclination and backfill slope. The lateral earth pressure formulation is based on Rankine's conjugate stress concept. The developed expression can be used for the static and pseudo-static seismic analyses of c–φ backfill. The results based on the proposed formulations are found to be identical to those computed with the Mononobe–Okabe method for cohesionless soils, provided the same wall friction angle is employed. For c–φ soils, the formulation yields comparable results to available solutions for cases where a comparison is feasible. The application eliminates the need for design charts, since it can accommodate any set of user-defined kinematically admissible design parameters.
Input parameters: The current version is working with S.I. units only. The input parameters for the example shown in Figure 5 are:
kh = 0.2
kv = -0.1
omega = 20°
beta = 15°
phi = 30°
upsilon = 23 kN/m³
c = 20 kPa
H = 15 m
"""


# In[5]:


import streamlit as st 
import openai
import json

from streamlit_chat import message
import numpy as np

from sep_core import sep

read_input = st.text_area("Enter ReadMe here:", height=25)
    
# Display the entered code
st.write("ReadMe:")
st.code(read_input)

code_input = st.text_area("Enter python function:", height=25)
    
# Display the entered code
st.write("Python Function:")
st.code(code_input, language="python")

openai.api_key = st.secrets["openai"]

def get_response_from_messages(messages, temperature, presence):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

def collect_messages(prompt): #add user's input, get chatgpt response
    st.session_state.context.append({'role':'user', 'content':f"{prompt}"})
    response = get_response_from_messages(st.session_state.context, 0.7, 0.6) 
    st.session_state.context.append({'role':'assistant', 'content':f"{response}"})
    st.session_state.past.append(prompt)
    st.session_state.generated.append(response)

make_prompt = [ {'role': 'system', 'content': """
I will give you information about a program. I want you to write a prompt for the prompt receiver to ask their user for information. Do not be the user or the program. Instruct the prompt receiver to gather information from the client to input into the program only. Include adequate knowledge in the prompt as the prompt receiver will have no knowledge of the program. Do not make assumptions about the program if it is not explicitly stated. Do not assume the range of a variable if not stated. Summarise information needed by the program into JSON format. Use the same variable name as the program needs in the JSON format, if specified. My explanations will be in < >, do not generate text in < >.

Program: 
I am a machine learning enthusiast who is currently working on classifying people's heart attack chance.
Using the Library
# enter valid values and calculate the risk.
>>> result = framingham_10year_risk(sex="male", age=26, total_cholesterol=152, hdl_cholesterol=70, systolic_blood_pressure=130, smoker=True, blood_pressure_med_treatment=False)
<Do not input the example values into the prompt>
Detecting Errors
result = framingham_10year_risk(sex="male-to-female-transgender", age=16, total_cholesterol=500, hdl_cholesterol=300, systolic_blood_pressure=20, smoker="foo", blood_pressure_med_treatment="bar")
>>> print result['errors']
['Blood pressure medication treatment must be set to True, False, 1 or 0.',
'Smoker must be set to True, False, 1, or 0.',
'Age must be within the range of 20 to 79.',
'Total cholesterol must be within the range of 130 to 320.',
'HDL cholesterol must be within the range of 20 to 100.',
'Systolic blood pressure must be within the range of 90 to 200 in mmHg.',
'Sex must be male or female.']

Additional - if the user does not know total cholesterol, HDL cholesterol or systolic blood pressure, substitute with age average. <Add additional needs to prompt if there are any>

Prompt: I want you to act like a nice, polite and helpful assistant. Ask me questions and wait for my answers. I will be the client and you are gathering my information to calculate my CVD risk. Ask me questions to do so, one by one. Do not move on to the next properties without knowing the value as stated in the range for the properties before it. Gather the information as said in the range. Do not calculate my risk percentage. I want you to only gather my information. Start by introducing me to the calculator, without introducing yourself, then start asking me for the following info one by one. Ask for my answer for the properties. The JSON format that they are in is {"property name": range (substitutable or not) (name to refer to property as and units), ...}: <Include the property or variable name in the JSON format. Include all information and criterias necessary to provide adequate knowledge in soft brackets.>
{"sex": female/male (N) (biological sex),
"age": 20-79 (N),
"total_cholesterol": 120-320 (Y),
"hdl_cholesterol": 20-100 (Y),
"systolic_blood_pressure": 80-200 (Y) (mmHg),
"current_smoker": 0/1 (N),
"blood_pressure_med_treatment": 0/1 (N) (taking blood pressure medicine)}
For properties that are substitutable (Y), if the user does not know their values for these properties, input the average value for their age instead. Ask me questions one by one as an assistant would, using varied and personal responses.


Program: 3rd grade History quiz
The history quiz is a mix of multiple choice and open-ended questions designed to test the knowledge of third graders. The third grader will be tested on the following questions:
When was the US founded?  
Who was the first president? George Washington/Thomas Jefferson 
How many stars are on the US flag? 
Is it true that Barack Obama is the 44th US president? 
The quiz program will then give out the percentage of correct answers based on the answers.
def HistoryQuiz(year_founded, first_president, stars_flag, Obama_44th_president):
    grade(year_founded, first_president, stars_flag, Obama_44th_president)

Additional- None
Prompt: I want you to act like a nice, polite and helpful assistant. I will be a third grade student and you will be asking for my history knowledge without correcting me. Only ask questions like someone examining the student. Ask me questions and wait for my answers. Do not calculate my score and do not correct my answer. Do not be me, the student or the human. Ask me questions to do so, one by one. Do not move on to the next properties without knowing the value as stated in the data type for the properties before it. Start by introducing me to the history quiz, without introducing yourself, then start asking me for the following info one by one. Ask for my answers for the question. The JSON format that they are in is {"question": data type or possible answers (question), ...}:
{"year_founded": integer (When was the US founded?),
"first_president": "George Washington"/"Thomas Jefferson" (Who was the first president?),
"stars_flag": integer (How many stars are on the US flag?),
"Obama_44th_president": boolean (Is it true that Barack Obama is the 44th US president?)}
Ask me questions one by one as a teacher would, using varied and personal responses.


Program: 
""" + user_program + "\nPrompt: "}]    


def check_for_risk():
   if "{" in st.session_state['generated'][-1]:
      last_message = st.session_state['generated'][-1]
      st.session_state.generated.pop()
   
      json_start = last_message.index('{')
      json_end = last_message.index('}')
      
      json_part = last_message[json_start:json_end+1]
      
      data = json.loads(json_part)
     
      try:
          result = sep(data['kh'], data['kv'], data['omega'], data['beta'], data['phi'], data['upsilon'], data['c'], data['H'])
          st.session_state.generated.append("result is " + str(result.z(9)))
      except (Exception) or (UnicodeDecodeError):
          find_error = [{'role': 'system', 'content': """
I will give you information about a program and the information that the user entered into the program in JSON format. You will identify the errors with the JSON that will not go into the program. To identify an error, check for any precedent for the information to see if the information that the user gave is plausible. If implausible, it is an error. If there is a precedent, it is not an error. Do not include Not Error when generating text.

Program: Framingham 10 Year Heart Attack Risk Calculator
Using the Library
# enter valid values and calculate the risk.
# Age must be between 20 and 70.
result = framingham_10year_risk(sex="male", age=26, total_cholesterol=152, hdl_cholesterol=70, systolic_blood_pressure=130, smoker=True, blood_pressure_med_treatment=False)
<Do not input the example values into the prompt>
User’s JSON: {“sex”: “non-binary”,
“age”: 15,
“total_cholesterol”: 190,
“hdl_cholesterol”: 170,
“systolic_blood_pressure”: 156,
“smoker”: “foo”,
“blood_pressure_med_ treatment”: True}
Not Error:
The user's total cholesterol is within normal range, so the program successfully takes it in and it is not an error.
The user's systolic blood pressure is higher than average but possible. So, it is not an error.
The blood pressure med treatment is given a correct boolean answer, so it is not an error.
Error:
The program asks for sex, which is biological sex, so the program can only take male or female.
The program said that age must be between 20 and 70, so the user’s age is outside of the program’s range.
The user’s HDL cholesterol is higher than normally possible. The user’s HDL cholesterol is not in the possible range.    
The program example for smoker is boolean. The user’s information is a string, which can’t be taken in by the program. The user needs to provide information on whether they are a smoker as a true or false.

Program: # Pipe pressure loss - Circular pipe, full flow water in SI Units
# based on Darcy-Weisbach, using Clamond algorithm for friction factor
# Inputs
D=70.3      # [mm]      Pipe diameter
aRou=0.1    # [mm]      Absolute roughness
mFlow=4.09  # [kg/s]    Mass flow rate
T=90        # [ºC]      Water temperature
L=66        # [m]       Pipe length
User’s JSON: {"D": 67, 
"aRou": 89, 
"mFlow": 2.5, 
"T": 150, 
"L": 10}
Not Error:
The pipe diameter, and pipe length is open-ended as long as it is a positive number, so it is not an error.
The mass flow rate is within normal range, so it is not an error.
Error:
The user’s absolute roughness is too high to be possible. The user’s absolute roughness is not in the possible range.
The user’s temperature is higher than possible for the water to flow through the pipe. The user’s temperature is not in the possible range.

Program: """ + user_program + "/nUser's JSON: " + json_part + "/nError: "}]
          fix_error = [{'role': 'system', 'content': "I want you to be a nice, polite and helpful assistant. You are asking the user for information to input into the program. The user's information in JSON format: \n" + json_part + "\nThe program found the following errors in the user's information: \n" + get_response_from_messages(find_error, 0,0) + "\nAsk the user to correct each error one by one. If you have all the errors corrected, substitute the error with the corrected information in the JSON. Use natural and service attitude with varied languages. If finished, generate the JSON format back to me without generating any other text. Do not make conversation with yourself. Do not ask about more than one error at a time. Do not ask about anything other than the specified error."}]

          st.session_state['context'] = fix_error[:]
          st.session_state.generated.append(get_response_from_messages(st.session_state['context'], 0.7, 0.6))

response_container = st.container()
input_container = st.container()

# Storing the chat
if 'context' not in st.session_state:
     prompt = [{'role': 'system', 'content': get_response_from_messages(make_prompt, 0, 0) + general}]
     st.session_state['context'] = prompt[:]

if 'generated' not in st.session_state:
    response = get_response_from_messages(st.session_state['context'], 0.7, 0)
    st.session_state['generated'] = [response]
    st.session_state.context.append({'role':'assistant', 'content':f"{response}"})

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.chat_input(placeholder="", key="input")
    return input_text

# Applying the user input box
with input_container:
    user_input = get_text()

with response_container:
    if user_input:
        collect_messages(user_input)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i), logo='https://www.freepnglogos.com/uploads/heart-png/emoji-heart-33.png')
            check_for_risk()


# In[ ]:




