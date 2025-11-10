from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()



def get_llm_response(prompt: str) -> str:
    """
    Generate a natural language response using google gemini via langchain. The model uses the provided context-rich prompt to generate the response.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        
        # model 
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # construct message for the model
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]
        # get response from the model
        response = llm.invoke(messages)
        return response.content.strip() # type: ignore
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return "I'm sorry, I couldn't process your request at this time."