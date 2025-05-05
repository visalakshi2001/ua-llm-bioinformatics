import streamlit as st
# from openai import OpenAI

from utilities import (initiate_client, stream_llm_reply, 
                       right_column_content, left_column_content)

st.set_page_config(page_title="Bioinformatics Lab Assistant",
                   page_icon="🧬",
                   layout="wide")

st.markdown(
    """
    <style>
    
    [data-testid="stMainBlockContainer"] {
            padding-top: 2%;
            padding-bottom: 0;
        }
    
    # /* Sticky page title */
    # .sticky-title {
    #     position: sticky;
    #     top: 0;
    #     z-index: 1000;
    #     background-color: var(--background-color);
    #     margin: 0;
    #     padding: 1.2rem 0 0.6rem 0;
    # }

    # /* Scrollable chat box */
    # #chat-scroll {
    #     height: 560px;
    #     overflow-y: auto;
    #     padding-right: 8px;  /* keep scrollbar off the text */
    # }

    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------  MAIN APP  ----------------------------------- #
def main() -> None:
    st.title("🧬 UA-LLM for bioinformatics research")

    # Session‑state initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello 👋 — how can I help you in the lab today?"}
        ]
    if "client" not in st.session_state:
        st.session_state.client = initiate_client()

    # --- Layout: chat (left) • uploads (right) ---
    left_col, right_col = st.columns([3, 1], gap="small")

    # ------------------  LEFT: chat area  ------------------ #
    with left_col:
        left_column_content()


    # ------------------  RIGHT: file‑upload area  ------------------ #
    with right_col:
        right_column_content()


if __name__ == "__main__":
    main()
