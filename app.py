import streamlit as st
import os
import sys
from langchain_together import ChatTogether
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from typing import List, Sequence
import shutil

# --- Configuration ---
# Add the parent directory to sys.path to find the vector_db folder
# Assuming app.py is in Chatbot_Rag and vector_db is parallel to Chatbot_Rag
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
PDF_DIRECTORY = os.path.join(current_dir, "documents")
VECTOR_DB_DIRECTORY = os.path.join(current_dir, "vector_db")
COLLECTION_NAME = "pdf"
# --- End Configuration ---


# --- Helper Functions (Caching expensive operations) ---
@st.cache_resource
def load_embeddings():
    """Load and cache the HuggingFace embeddings model."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        st.sidebar.success("Embeddings loaded successfully.")
        return embeddings
    except Exception as e:
        st.error(f"Fatal Error: Could not load embeddings model: {e}")
        st.sidebar.error("Embeddings failed to load.")
        return None

@st.cache_resource
def get_vector_store(_embeddings):
    """Loads an existing vector store or creates a new one from PDFs if necessary.
    Also synchronizes the existing store with the PDF_DIRECTORY."""
    if not _embeddings:
        st.error("Embeddings not available. Cannot get vector store.")
        st.sidebar.error("Embeddings not available.")
        return None

    # Helper function to load, split, and add new documents
    def add_new_documents(vector_store, file_paths, pdf_directory):
        added_docs = []
        st.sidebar.info(f"Found {len(file_paths)} new PDF(s) to add...")
        progress_bar_add = st.sidebar.progress(0, text="Adding new PDFs...")
        new_docs_count = 0
        new_chunks_count = 0
        for i, file_path_full in enumerate(file_paths):
            pdf_file_name = os.path.basename(file_path_full)
            try:
                loader = PyPDFLoader(file_path_full)
                docs = loader.load()
                if not docs:
                    st.sidebar.warning(f"No content loaded from {pdf_file_name}. Skipping.")
                    continue
                for doc in docs:
                    doc.metadata["source"] = pdf_file_name # Ensure source metadata is set
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                if splits:
                    vector_store.add_documents(splits)
                    new_docs_count += 1
                    new_chunks_count += len(splits)
                    added_docs.append(pdf_file_name)
                else:
                     st.sidebar.warning(f"No text chunks generated for {pdf_file_name} after splitting.")
            except Exception as e:
                st.sidebar.error(f"Error processing new file {pdf_file_name}: {e}")
            progress_bar_add.progress((i + 1) / len(file_paths), text=f"Adding {pdf_file_name}...")
        progress_bar_add.empty()
        if new_docs_count > 0:
             st.sidebar.success(f"Added {new_docs_count} doc(s) ({new_chunks_count} chunks): {', '.join(added_docs)}")
        return new_docs_count > 0 # Return True if changes were made


    # Helper function to delete documents associated with removed files
    def remove_deleted_documents(vector_store, deleted_files):
        if not deleted_files:
            return False # No changes made

        st.sidebar.info(f"Found {len(deleted_files)} removed PDF(s). Removing from store...")
        removed_files_str = ', '.join(deleted_files)
        st.sidebar.write(f"Files to remove: {removed_files_str}")

        # Find document IDs associated with the deleted files
        ids_to_delete = []
        retrieved_docs = vector_store.get(include=["metadatas"]) # Get all docs
        all_ids = retrieved_docs.get("ids", [])
        all_metadatas = retrieved_docs.get("metadatas", [])

        if not all_ids or not all_metadatas:
             st.sidebar.warning("Could not retrieve IDs/metadata for deletion.")
             return False

        for doc_id, metadata in zip(all_ids, all_metadatas):
            if metadata and metadata.get("source") in deleted_files:
                ids_to_delete.append(doc_id)

        if ids_to_delete:
            try:
                st.sidebar.write(f"Deleting {len(ids_to_delete)} chunks...")
                vector_store.delete(ids=ids_to_delete)
                st.sidebar.success(f"Removed chunks for: {removed_files_str}")
                return True # Return True if changes were made
            except Exception as e:
                st.sidebar.error(f"Error deleting documents: {e}")
                return False # Changes were attempted but failed
        else:
            st.sidebar.info("No chunks found for removed files.")
            return False # No changes made

    # --- Attempt 1: Load existing Vector Store ---
    vector_store = None
    if os.path.exists(VECTOR_DB_DIRECTORY) and os.listdir(VECTOR_DB_DIRECTORY):
        st.sidebar.info(f"Loading vector store...")
        try:
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=_embeddings,
                persist_directory=VECTOR_DB_DIRECTORY
            )
            # Perform a small operation to check if it's valid
            _ = vector_store.get(limit=1)
            st.sidebar.success("Vector store loaded successfully.")

            # --- Synchronization Logic ---
            st.sidebar.info("Checking for document updates...")
            if not os.path.isdir(PDF_DIRECTORY):
                 st.sidebar.warning(f"PDF directory not found. Cannot sync.")
            else:
                # 1. Get current files in the PDF directory
                current_pdf_files = set(f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf"))
                st.sidebar.write(f"Docs folder files: {len(current_pdf_files)}")

                # 2. Get document sources from the vector store
                stored_docs_data = vector_store.get(include=["metadatas"])
                stored_sources = set()
                if stored_docs_data and stored_docs_data.get("metadatas"):
                     for meta in stored_docs_data["metadatas"]:
                         if meta and "source" in meta:
                              stored_sources.add(meta["source"])
                st.sidebar.write(f"Indexed files: {len(stored_sources)}")

                # 3. Identify files to add and delete
                files_to_add_names = current_pdf_files - stored_sources
                files_to_delete_names = stored_sources - current_pdf_files

                files_to_add_paths = [os.path.join(PDF_DIRECTORY, fname) for fname in files_to_add_names]

                # 4. Add new documents
                if files_to_add_paths:
                    add_new_documents(vector_store, files_to_add_paths, PDF_DIRECTORY)

                # 5. Remove deleted documents
                if files_to_delete_names:
                     remove_deleted_documents(vector_store, list(files_to_delete_names))

                if not files_to_add_names and not files_to_delete_names:
                    st.sidebar.info("Vector store is up-to-date.")

            # --- End Synchronization Logic ---

        except Exception as e:
            st.sidebar.warning(f"Load/sync failed: {e}. Rebuilding...")
            vector_store = None # Reset vector_store to trigger rebuild
            try:
                 shutil.rmtree(VECTOR_DB_DIRECTORY)
                 os.makedirs(VECTOR_DB_DIRECTORY)
            except Exception as cleanup_e:
                 st.sidebar.error(f"Failed cleanup: {cleanup_e}")
                 # Proceed to try loading documents anyway, maybe it works without cleanup

    # --- Attempt 2: Create Vector Store from Documents (if loading failed or dir empty/missing) ---
    if vector_store is None: # If loading failed or store didn't exist
        st.sidebar.info("Creating new vector store from PDFs...")
        all_docs = []
        if not os.path.isdir(PDF_DIRECTORY):
            st.error("Failed to load/find existing vector store and no PDF directory found to create a new one.")
            st.sidebar.error("PDF dir not found. Cannot create store.")
            return None # Critical failure
        else:
            pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
            if not pdf_files:
                st.error("Failed to load/find existing vector store and no PDF files found to create a new one.")
                st.sidebar.error("No PDFs found. Cannot create store.")
                return None # Critical failure
            else:
                # Load PDFs
                progress_bar_load = st.sidebar.progress(0, text="Loading PDFs...")
                successful_loads = 0
                for i, pdf_file in enumerate(pdf_files):
                    file_path = os.path.join(PDF_DIRECTORY, pdf_file)
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["source"] = pdf_file # Set source metadata
                        all_docs.extend(docs)
                        successful_loads += 1
                    except Exception as e:
                        st.sidebar.error(f"Error loading {pdf_file}: {e}")
                    progress_bar_load.progress((i + 1) / len(pdf_files), text=f"Loading {pdf_file}...")
                progress_bar_load.empty()

                if not all_docs:
                    st.error("No documents were successfully loaded from PDF folder. Cannot create vector store.")
                    st.sidebar.error("No documents loaded. Cannot create store.")
                    return None # Critical failure

                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                all_splits = text_splitter.split_documents(all_docs)
                st.sidebar.info(f"Processing {successful_loads} PDFs into {len(all_splits)} chunks...")

                # Create and persist the new vector store
                try:
                    os.makedirs(VECTOR_DB_DIRECTORY, exist_ok=True)
                    # Chroma.from_documents automatically persists if persist_directory is provided
                    vector_store = Chroma.from_documents(
                        documents=all_splits,
                        embedding=_embeddings,
                        collection_name=COLLECTION_NAME,
                        persist_directory=VECTOR_DB_DIRECTORY
                    )
                    st.sidebar.success(f"New vector store created.")
                    # No need to explicitly persist here, from_documents does it.
                except Exception as e:
                    st.error(f"Error creating vector store from documents: {e}")
                    st.sidebar.error(f"Error creating vector store: {e}")
                    return None # Critical failure

    return vector_store # Return the potentially updated or newly created store

# --- LangGraph Setup ---

# The actual implementation function (renamed)
def _execute_retrieval(query: str, vector_store: Chroma):
    """Actual implementation to retrieve information."""
    if not vector_store:
        return "Vector store is not available."
    try:
        retrieved_docs = vector_store.similarity_search(query, k=3) # Retrieve 3 chunks
        if not retrieved_docs:
            return "No relevant information found in the documents."
        serialized = "\n\n".join(
            (f"Source: {doc.metadata.get('source', 'N/A')} - Page: {doc.metadata.get('page', 'N/A')}\n"
             f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        return f"Error during retrieval: {e}"

# Tool definition stub for the LLM (only includes arguments LLM needs to provide)
@tool
def retrieve(query: str):
    """Retrieve information related to the query from loaded documents."""
    # NOTE: The actual implementation is in _execute_retrieval and called by CustomToolNode.
    # This stub is just for schema generation for the LLM via bind_tools.
    pass

def get_chatbot_graph(api_key: str, temperature: float, vector_store: Chroma):
    """Builds and compiles the LangGraph agent."""
    if not api_key:
        st.error("Together API Key is missing.")
        st.sidebar.error("API Key missing.")
        return None
    if not vector_store:
        st.error("Vector Store is not available. Cannot build graph.")
        st.sidebar.error("Vector Store unavailable.")
        return None

    try:
        llm = ChatTogether(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-classifier",
            temperature=temperature,
            api_key=api_key
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM. Check API Key and model name: {e}")
        st.sidebar.error("LLM initialization failed.")
        return None

    # --- Graph Nodes ---

    def query_or_respond_node(state: MessagesState):
        """Decide whether to use tool or respond directly."""
        # Bind the STUB tool definition, not the implementation with Chroma object
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # CustomToolNode executes the *actual* implementation (_execute_retrieval)
    class CustomToolNode:
        def __init__(self, vs: Chroma):
            self.vs = vs
            if not vs:
                # Log this issue during initialization if possible
                print("Warning: CustomToolNode initialized without a valid vector store.")

        def __call__(self, state: MessagesState):
            if not self.vs:
                # Handle the case where vector store wasn't loaded properly
                st.error("Tool call failed: Vector store not available.")
                # Return a ToolMessage indicating the failure
                tool_invocation = state["messages"][-1].tool_calls[0] # Assume tool call exists
                return {
                    "messages": [
                        ToolMessage(content="Error: Vector store could not be loaded.", tool_call_id=tool_invocation["id"])
                    ]
                }

            # Find the tool call requested by the LLM
            tool_invocation = None
            if state["messages"][-1].tool_calls:
                tool_invocation = state["messages"][-1].tool_calls[0] # Assuming one tool call

            if not tool_invocation:
                # This path might be hit if the conditional edge logic changes
                # Return an empty AI message or handle as appropriate
                return {"messages": [AIMessage(content="No tool was called.")]}

            # Get the tool name from the tool call
            tool_name = tool_invocation.get("name", "")
            if not tool_name:
                 # If name is missing, return error ToolMessage
                return {"messages": [ToolMessage(content="Error: Tool name missing in invocation.", tool_call_id=tool_invocation["id"])]}

            # Check if the called tool is our 'retrieve' tool
            if tool_name != "retrieve":
                 return {"messages": [ToolMessage(content=f"Error: Unknown tool '{tool_name}' called.", tool_call_id=tool_invocation["id"])]}

            # Call the *actual* implementation function with the vector store
            try:
                tool_output = _execute_retrieval(
                    query=tool_invocation["args"]["query"],
                    vector_store=self.vs
                )
                return {
                    "messages": [
                        ToolMessage(content=str(tool_output), tool_call_id=tool_invocation["id"])
                    ]
                }
            except Exception as node_e:
                # Catch errors during retrieval execution within the node
                st.error(f"Error executing retrieval tool: {node_e}")
                return {"messages": [ToolMessage(content=f"Error executing tool: {node_e}", tool_call_id=tool_invocation["id"])]}

    # Instantiate CustomToolNode safely
    custom_tool_node = CustomToolNode(vector_store) if vector_store else None
    if not custom_tool_node:
         st.sidebar.warning("Retrieval tool disabled (Vector Store missing).")

    def generate_node(state: MessagesState):
        """Generate final response based on context and history."""
        # Get the most recent tool message content
        tool_messages = [m for m in reversed(state["messages"]) if isinstance(m, ToolMessage)]
        docs_content = "\n\n".join(
            [str(m.content) for m in tool_messages]
        ) if tool_messages else "No information retrieved or tool failed."

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer from the context, or if context indicates an error or no information, say that you "
            "don't know or report the issue concisely. Use three sentences maximum and keep the "
            "answer concise. DO NOT mention the tool call unless specifically asked."
            "\n\nCONTEXT:\n"
            f"{docs_content}"
        )

        # Filter for non-tool AI messages and human messages for conversation context
        conversation_messages = [
            m for m in state["messages"] if isinstance(m, HumanMessage) or
            (isinstance(m, AIMessage) and not m.tool_calls)
        ]

        prompt_messages = [SystemMessage(content=system_prompt)] + conversation_messages

        try:
            response = llm.invoke(prompt_messages)
            return {"messages": [response]}
        except Exception as gen_e:
            st.error(f"Error during response generation: {gen_e}")
            # Return an error message as AIMessage
            return {"messages": [AIMessage(content=f"Sorry, an error occurred while generating the response: {gen_e}")]}

    # --- Define Graph ---
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("query_or_respond", query_or_respond_node)
    # Only add the tool node if it was successfully created
    if custom_tool_node:
        graph_builder.add_node("call_tool", custom_tool_node)
    else:
        # If no tool node, create a dummy node that passes through or indicates failure
        def failed_tool_node(state: MessagesState):
             tool_invocation = state["messages"][-1].tool_calls[0] # Assume tool call exists
             return {"messages": [ToolMessage(content="Error: Retrieval tool not available (Vector Store missing).", tool_call_id=tool_invocation["id"])]}
        graph_builder.add_node("call_tool", failed_tool_node) # Add dummy/error node

    graph_builder.add_node("generate", generate_node)

    graph_builder.set_entry_point("query_or_respond")

    # Conditional Edges: Check if the LLM response contains tool calls
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition, # Checks messages[-1].tool_calls
        {
            "tools": "call_tool", # Route to tool handling (real or dummy)
            END: "generate" # Route to generation if no tool call needed
        }
    )
    # Always route from tool result/failure back to generation
    graph_builder.add_edge("call_tool", "generate")
    graph_builder.add_edge("generate", END)

    # Compile with memory
    memory = MemorySaver()
    try:
        graph = graph_builder.compile(checkpointer=memory)
        st.sidebar.success("Chatbot graph compiled.")
        return graph
    except Exception as e:
        st.error(f"Error compiling graph: {e}")
        st.sidebar.error("Graph compilation failed.")
        return None

# --- Streamlit App UI ---

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

st.title("📚 PDF RAG Chatbot")
st.caption("Chat with your PDF documents using LangGraph and Together AI")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input(
        "Together AI API Key",
        type="password",
        key="api_key_input",
        help="Get yours from https://api.together.xyz/",
        value=st.session_state.get("api_key", "") # Persist in session state
    )
    # Store API key when entered
    if api_key_input:
        st.session_state.api_key = api_key_input

    temperature_input = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("temperature", 0.0), # Persist temperature
        step=0.1,
        key="temperature_slider"
    )
    # Store temperature when changed
    st.session_state.temperature = temperature_input

    st.divider()
    st.caption(f"PDF Directory (for new DB): {PDF_DIRECTORY}")
    st.caption(f"Vector DB Directory: {VECTOR_DB_DIRECTORY}")
    st.caption(f"Collection: {COLLECTION_NAME}")


# --- Main Chat Interface ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load resources only if API key is provided
if st.session_state.get("api_key"):
    # Load embeddings first (critical)
    embeddings = load_embeddings()

    if embeddings:
        # Attempt to get the vector store (load or create)
        vector_store = get_vector_store(embeddings)

        if vector_store:
            # Display chat messages from history
            for message in st.session_state.messages:
                avatar = "👤" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

            # Get the compiled graph
            chatbot_graph = get_chatbot_graph(
                st.session_state.api_key,
                st.session_state.temperature,
                vector_store
            )

            # React to user input
            if prompt := st.chat_input("Ask a question about your documents..."):
                if not chatbot_graph:
                    st.error("Chatbot graph could not be initialized. Cannot process query.")
                else:
                    # Add user message to chat history and display
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(prompt)

                    # Prepare graph input from history
                    graph_input_messages = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            graph_input_messages.append(HumanMessage(content=msg["content"]))
                        else: # Assuming role is 'assistant'
                            graph_input_messages.append(AIMessage(content=msg["content"]))


                    # Unique ID for the conversation thread
                    config = {"configurable": {"thread_id": "streamlit_chat_1"}}

                    # Stream the response
                    with st.chat_message("assistant", avatar="🤖"):
                        message_placeholder = st.empty()
                        full_response = ""
                        final_ai_message_content = None # Store only the final AI response content


                        try:
                            events = chatbot_graph.stream(
                                {"messages": graph_input_messages},
                                config=config,
                                stream_mode="values"
                            )

                            for step_output in events:
                                # Get the last message added in this step
                                last_message = step_output["messages"][-1]

                                # Display intermediate steps? (Optional - can be verbose)
                                # print(f"Step Output: {last_message.pretty_repr()}") # For debugging

                                # Update UI only with the *final* assistant message
                                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                                     # This assumes the final message is the last AIMessage without tool calls
                                    final_ai_message_content = last_message.content
                                    message_placeholder.markdown(final_ai_message_content + "▌")

                            # Final update after loop finishes
                            if final_ai_message_content is not None:
                                message_placeholder.markdown(final_ai_message_content)
                                full_response = final_ai_message_content
                            else:
                                # Handle cases where no final AIMessage was found (e.g., graph error before generation)
                                full_response = "Error: Could not determine the final response."
                                message_placeholder.error(full_response)


                        except Exception as e:
                            full_response = f"An error occurred during graph execution: {e}"
                            st.error(full_response)
                            # Add error to history? Might be confusing.
                            # st.session_state.messages.append({"role": "assistant", "content": full_response})


                    # Add final assistant response to chat history
                    if full_response and not full_response.startswith("An error occurred"):
                         st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
             st.warning("Vector Store could not be loaded or created. Chatbot is not available.")
    else:
        st.error("Embeddings model failed to load. Chatbot cannot start.")

else:
    st.info("Please enter your Together AI API Key in the sidebar to start the chatbot.")
