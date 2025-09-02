import streamlit as st
from langchain.memory import ConversationBufferMemory
from pdf问答_h import ai_agent

st.title('pdf问答🗒️')
with st.sidebar:
    api_key=st.text_input('请输入api秘钥',type='password')
    base_url=st.text_input('请输入base_url')

if 'memory' not in st.session_state:
    st.session_state['memory']=ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key='answer'
    )
up_loader=st.file_uploader('上传你的PDF文件',type='pdf')
#disabled:当什么时候不允许输入
question=st.text_input('请提问',disabled=not up_loader)
if question:
    if not api_key:
        st.info('请输入你的api秘钥')
        st.stop()
    with st.spinner('AI正在思考中,请稍等...'):
        response=ai_agent('sk-UmDxWlpFiAqen1tz0GU6SAadJFTdcaIC9KwbJppia7F9CXoD','https://www.chataiapi.com/v1',st.session_state['memory'],up_loader,question)
    st.write('### 答案')
    st.write(response['answer'])
    st.session_state['chat_history']=response['chat_history']

if 'chat_history' in st.session_state:
    with st.expander("历史消息"):
        for i in range(0,len(st.session_state["chat_history"]),2):
            human_message=st.session_state["chat_history"][i]
            ai_message=st.session_state['chat_history'][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i <len(st.session_state['chat_history'])-2:
                st.divider()