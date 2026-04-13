from langchain.tools import tool

def get_search_knowledge_base_tool(rag_system):
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the IT support knowledge base for relevant documents.
        Use this tool when the user asks any IT-related question about
        passwords, VPN, software, hardware, printers, network, etc.
        Pass the user's question or relevant keywords as the query.
        """
        # Switch retrieval strategy based on current mode
        mode = getattr(rag_system, "_retrieval_mode", "hybrid")
        if mode == "dense":
            docs = rag_system.dense_retriever.invoke(query)[:3]
        else:
            docs = rag_system._hybrid_retrieve(query, top_k=3)

        if not docs:
            return "No relevant documents found in the knowledge base."

        results = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            results.append(f"[{source}]: {doc.page_content}")

        # Store sources for later extraction
        rag_system._last_sources = [
            {
                "source": doc.metadata.get("source", "Unknown"), 
                "content": doc.page_content,
                "metadata": doc.metadata,
                "text": doc.page_content
            }
            for doc in docs
        ]

        return "\n\n".join(results)

    return search_knowledge_base
