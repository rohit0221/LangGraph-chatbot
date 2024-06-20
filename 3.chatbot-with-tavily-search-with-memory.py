from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
import subprocess
# Import Libraries
from langchain_openai import ChatOpenAI

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
import json
from typing import Literal
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")


# Initialize the ChatOpenAI model 
llm = ChatOpenAI(model="gpt-3.5-turbo")
# Create tools
tool = TavilySearchResults(max_results=2)
tools = [tool]

#Create LLM with tools
llm_with_tools = llm.bind_tools(tools)

# Create the State Class
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them
class State(TypedDict):
    messages: Annotated[list, add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    

#Class to manage ChatBot. Takes LLM as parameter
class ChatbotManager:
    def __init__(self, model):
        #Assign the model to an instance variable called llm
        self.llm = model


        #Create an instance of Class StateGraph with an initial state called State.
        self.graph_builder = StateGraph(State)
        # Add a node named “chatbot” to the state graph, associated with a function called self.chatbot
        tool_node = BasicToolNode(tools=[tool])
        self.graph_builder.add_node("chatbot", self.chatbot)
        self.graph_builder.add_node("tools", tool_node)

        # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
        # it is fine directly responding. This conditional routing defines the main agent loop.

        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        self.graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        # Any time a tool is called, we return to the chatbot to decide the next step
        self.graph_builder.add_edge("tools", "chatbot")
        self.graph_builder.set_entry_point("chatbot")
        self.graph = self.graph_builder.compile(checkpointer=memory)

    # Define the node now. Node is just a function that does some work.  Nodes represent units of work. They are typically regular python functions.  
    # in this case the node is a Chatbot that answers the questions.
    # Every node node we define will receive the current State as input and return a value that updates that state
    # This fun ction takes input parameter "state" of type "State".
    # processes the current state and generates a response
    # state["messages"] represents the input messages or conversation history stored in the state object.
    # The method invokes the language model with the input messages and returns a dictionary with a single key-value pair:
    # Key: "messages"
    # Value: A list containing the chatbot’s response(s).

    def chatbot(self, state: State):
        return {"messages": [self.llm.invoke(state["messages"])]}

    def display_graph(self):
        try:
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception as e:
            print(f"Graph display error: {e}")
    def save_graph_as_image(self, filename="graph.png"):
        try:
            graph_image = self.graph.get_graph().draw_mermaid_png()
            with open(filename, "wb") as img_file:
                img_file.write(graph_image)
            print(f"Graph saved as {filename}")
        except Exception as e:
            print(f"Graph save error: {e}")


    def open_graph_with_viewer(self, filename="graph.png"):
        try:
            graph_image = self.graph.get_graph().draw_mermaid_png()
            with open(filename, "wb") as img_file:
                img_file.write(graph_image)
            print(f"Graph saved as {filename}")
            # Open the image with the default viewer (Windows)
            subprocess.run(["start", filename], shell=True)
        except Exception as e:
            print(f"Graph save and open error: {e}")

    def run(self):
        while True:
            config = {"configurable": {"thread_id": "1"}}
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            #iterate over events from the state graph.
            #The {"messages": ("user", user_input)} dictionary represents the current state with the user’s input message.

            # The config is the **second positional argument** to stream() or invoke()!
            events = self.graph.stream(
                {"messages": [("user", user_input)]}, config, stream_mode="values"
            )
            for event in events:
                event["messages"][-1].pretty_print()



            # for event in self.graph.stream({"messages": ("user", user_input)},config,stream_mode="values"):
            #     # This inner loop extracts values from the event
            #     for value in event.values():
            #         if isinstance(value["messages"][-1], BaseMessage):
            #             # retrieve the content of the last message generated by the chatbot.
            #             print("Assistant:", value["messages"][-1].content)



if __name__ == "__main__":
    chatbot_manager = ChatbotManager(llm_with_tools)
    
    chatbot_manager.display_graph()
    chatbot_manager.save_graph_as_image()
    #chatbot_manager.open_graph_with_viewer()
    chatbot_manager.run()
