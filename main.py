from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool

import streamlit as st
from streamlit_chat import message

load_dotenv()

st.set_page_config(page_title="Agente Inteligente", layout="centered")
st.header("Agente Inteligente - Python y CSV")


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have acceess to a python REPL, which you can use to execute python code.
    You ahve qrcode package installed.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get to the answer.
    If it does not seem like to you can write code to answer the question, just return "I don't know" as the answer. """

    base_promt = hub.pull("langchain-ai/react-agent-template")
    tools = [PythonREPLTool()]


    prompt = base_promt.partial(instructions=instructions)
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools
    )
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    csv_files = ["episode_info.csv"] 
    csv_agents = {
        file: create_csv_agent(
            llm=ChatOpenAI(temperature=0, model="gpt-4"),
            path=file,
            verbose=True,
            allow_dangerous_code=True,
        ) for file in csv_files
    }

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor.invoke,
            description="""useful when you need to transform natural language to python and execute the python code,
            returning the results of the code execution
            DOES NOT ACCEPT CODE AS INPUT""",
        )
    ]

    tools += [
        Tool(
         name=f"CSV Agent ({csv})",
         func=csv_agents[csv].invoke,   
         description=f"Useful for answering questions related to {csv}.",
        ) for csv in csv_files
    ]

    master_agent = create_react_agent(
        prompt=base_promt.partial(instructions=""),
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    master_agent_executor = AgentExecutor(agent=master_agent, tools=tools, verbose=True)

    st.sidebar.header("Menú de selección")
    task = st.sidebar.selectbox(
        "Selecciona una tarea para el Python Agent:",
        ["Generar QRCodes", "Calcular suma de números", "Graficar datos simples"]
    )

    if st.sidebar.button("Ejecutar tarea"):
        if task == "Generar QRCodes":
            input_text = "Generate and save in current working directory 15 qrcodes that point to www.google.com"
        elif task == "Calcular suma de números":
            input_text = "Write a Python program that calculates the sum of numbers from 1 to 100."
        elif task == "Graficar datos simples":
            input_text = "Generate a Python program to plot y=x^2 for x in range -10 to 10."
        
        response = python_agent_executor.invoke({"input": input_text})
        st.text(f"Resultado de la tarea seleccionada:\n{response['output']}")

    st.subheader("Pregunta para los CSVs o Python Agent")
    user_input = st.text_area("Escribe tu pregunta aquí:", height=100)

    if st.button("Procesar pregunta"):
        if user_input.strip():
            response = master_agent_executor.invoke({"input": user_input})
            st.text(f"Respuesta del agente:\n{response['output']}")
        else:
            st.warning("Por favor, escribe una pregunta válida.")

if __name__ == '__main__':
    main()


