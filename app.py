from langchain_aws import ChatBedrock
from langchain.chains import ConversationChain, MultiPromptChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
import datetime
from langchain.chains.llm import LLMChain
import gradio as gr


class PromptFactory():


    current_date = datetime.datetime.now()
    summarizer_template = """
    Summarize the following patient-reported medical information concisely in 3rd person: {input}. """ + \
    f"Specify the exact date if a relative time is mentioned, based on {current_date}. " + \
    """At the end, if the patient has thyroid cancer or a family history of it, inform them they can't take semaglutide.

    Here is a question:
    {input}"""

    asker_template = """
    You are a helpful assistant.
    You have been given the following patient-reported data: {input}. 
    Ask up to 3 follow up questions to find out as much medical information as possible.
    Also, ask the patient if they have a family history of thyroid cancer.
    If the patient has thyroid cancer or a family history of it, inform them they cannot take semaglutide.

    Here is a question:
    {input}"""
  
    advisor_template = """You are a healthcare expert. \
    You answer common knowledge questions based on your vast knowledge.
    Your explanations are concise.

    Here is a question:
    {input}"""

    legal_expert_template = """You are a US healthcare legal expert. \
    You explain questions related to the US legal system with a good number of examples.

    Here is a question:
    {input}"""



    prompt_infos = [
        {
            'name': 'Summarizer',
            'description': 'Good for summarizing medical information',
            'prompt_template': summarizer_template
        },
        {
            'name': 'Asker',
            'description': 'Good for extracting more information from the patient',
            'prompt_template': asker_template
        },
        {
            'name': 'Advisor',
            'description': 'Good for telling the patient best course of action',
            'prompt_template': advisor_template
        },

        {
            'name': 'legal expert',
            'description': 'Good for answering questions which are related to UK or US law',
            'prompt_template': legal_expert_template
        }
    ]




def generate_destination_chains():
    """
    Creates a list of LLM chains with different prompt templates.
    """
    prompt_factory = PromptFactory()
    destination_chains = {}
    for p_info in prompt_factory.prompt_infos:
        name = p_info['name']
        prompt_template = p_info['prompt_template']
        chain = LLMChain(
            llm=llm, 
            memory=memory,
            prompt=PromptTemplate(template=prompt_template, input_variables=['input']))
        destination_chains[name] = chain
    default_chain = ConversationChain(llm=llm, memory=memory, output_key="text")
    return prompt_factory.prompt_infos, destination_chains, default_chain

def generate_router_chain(prompt_infos, destination_chains, default_chain):
    """
    Generats the router chains from the prompt infos.
    :param prompt_infos The prompt informations generated above.
    """
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = '\n'.join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=['input'],
        output_parser=RouterOutputParser()
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    return MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
        #callbacks=[file_ballback_handler]
    )
    

#to let model know current day
current_date = datetime.datetime.now()
current_day = current_date.day


memory = ConversationBufferMemory()

llm = ChatBedrock(credentials_profile_name = 'default',
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        streaming=True,
        model_kwargs=dict(temperature=0),
        # other params...
    )

prompt_infos, destination_chains, default_chain = generate_destination_chains()
chain = generate_router_chain(prompt_infos, destination_chains, default_chain)
    

 

# Gradio Function
def chatbot(message):
    if message.lower() in ["restart", "reset"]:
        memory.clear()  # Clear memory on reset
        return "Conversation has been restarted. Hi, new patient. Can you provide your medical information?"
    response = chain.run(input=message)
    return response

with gr.Blocks() as demo:
    gr.Markdown("# 💊 Healthcare Chatbot")
    gr.Markdown("#### Provide your medical information and receive a summary and medical suggestions.")

    with gr.Tab("Query"):
        with gr.Column():
            response_output = gr.Markdown(elem_classes="gr-markdown")
            query_input = gr.Textbox(label="Query", placeholder="Enter your medical info here...", 
                                     elem_classes="gr-textbox")
            submit_button = gr.Button("Submit", elem_classes="gr-button")
        
        submit_button.click(fn=chatbot, 
                            inputs=[query_input], 
                            outputs=response_output)

    with gr.Tab("About"):
        gr.Markdown("### About This Chatbot")
        gr.Markdown("Type reset to restart the conversation.")

demo.launch(share=True)