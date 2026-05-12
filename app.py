import gradio as gr

from agent import create_chat_runnable
from memory import search_video_memory
from video_processing import process_video


def build_ui() -> gr.Blocks:
    chat_llm = create_chat_runnable()

    with gr.Blocks(title="CamQuery Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎥 CamQuery Assistant")
        gr.Markdown(
            "Upload videos, process them, then ask questions about content, timestamps, objects, and scenes!"
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_upload = gr.File(
                    label="📹 Upload Video",
                    file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    type="filepath",
                )
                process_btn = gr.Button("🔄 Process Video", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500)
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about video content...",
                        label="💬 Your Question",
                        scale=3,
                    )
                    send_btn = gr.Button("Send", scale=1)
                clear_btn = gr.Button("🗑️ Clear Chat")

        def process_uploaded_video(video_path: str) -> str:
            if not video_path:
                return "❌ No video uploaded!"

            try:
                return f"✅ {process_video(video_path)}"
            except Exception as error:
                return f"❌ Error: {error}"

        def respond(message: str, history: list[dict]) -> tuple[str, list[dict]]:
            if not message.strip():
                return "", history

            try:
                memory = search_video_memory(message)
                recent_history = history[-6:]
                conversation_context = ""

                for item in recent_history:
                    role = item.get("role", "user").capitalize()
                    conversation_context += f"{role}: {item.get('content', '')}\n"

                raw_response = chat_llm.invoke(
                    {
                        "message": message,
                        "conversation_context": history,
                        "memory": memory,
                    }
                )

                response = raw_response["messages"][-1]
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response.content})
                return "", history
            except Exception as error:
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"❌ Error: {error}"})
                return "", history

        def clear_chat() -> tuple[list, str]:
            return [], ""

        process_btn.click(process_uploaded_video, inputs=[video_upload], outputs=[status])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        send_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear_btn.click(clear_chat, outputs=[chatbot, msg])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True,
    )

