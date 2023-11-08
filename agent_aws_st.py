# Import libraries
import streamlit as st
import aws_langchain_rag_demo_redacted


@st.cache_resource

def sidebar() -> None:
    """
    Purpose:
        Shows the side bar
    Args:
        N/A
    Returns:
        N/A
    """

    st.sidebar.image(
        "https://d1.awsstatic.com/gamedev/Programs/OnRamp/gt-well-architected.4234ac16be6435d0ddd4ca693ea08106bc33de9f.png",
        use_column_width=True,
    )

    st.sidebar.markdown(
        "AWS Genie is an intelligent chatbot that uses Amazon Sagemaker, Amazon Bedrock and Langchain"
    )


def app() -> None:
    """
    Purpose:
        Controls the app flow
    Args:
        N/A
    Returns:
        N/A
    """

    # Spin up the sidebar
    sidebar()

    query = st.text_input("Query:")

    if st.button("Submit Query"):
        with st.spinner("Generating..."):
            response = aws_langchain_rag_demo_redacted.query(query)
            reply = response['result']
            if not aws_langchain_rag_demo_redacted.check_no_source(reply):
                reply += "\n\nSource:\n"
                urls=aws_langchain_rag_demo_redacted.find_metadata_sources_from_documents(response['source_documents'])
                for url in urls:
                    reply += "- "+url+"\n"
            st.markdown(reply)


def main() -> None:
    """
    Purpose:
        Controls the flow of the streamlit app
    Args:
        N/A
    Returns:
        N/A
    """

    # Start the streamlit app
    st.title("AWS Genie")
    st.subheader("Ask Genie not for wishes but for answers!")

    app()


if __name__ == "__main__":
    main()
