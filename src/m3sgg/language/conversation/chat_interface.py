"""
Scene Graph Chat Interface

This module provides a high-level chat interface for integrating LLM-based
conversation capabilities with scene graph analysis in Streamlit applications.

Classes:
    SceneGraphChatInterface: Main chat interface for scene graph conversations
"""

import logging
from typing import Any, Dict
import streamlit as st
from streamlit_chat import message
import uuid

from ..language_modeling.llm import (
    create_conversation_manager,
    SceneGraphFormatter
)

logger = logging.getLogger(__name__)


class SceneGraphChatInterface:
    """Main chat interface for scene graph conversations.
    
    Provides a complete chat interface that integrates with Streamlit
    and handles scene graph context for natural language conversations.
    
    :param model_name: Name of the LLM model to use, defaults to "google/gemma-3-270m"
    :type model_name: str, optional
    :param model_type: Type of model wrapper, defaults to "gemma"
    :type model_type: str, optional
    :param max_context_length: Maximum conversation context length, defaults to 10
    :type max_context_length: int, optional
    """
    
    def __init__(self, model_name: str = "google/gemma-3-270m", 
                 model_type: str = "gemma", max_context_length: int = 10):
        """Initialize the chat interface.
        
        :param model_name: Name of the LLM model to use, defaults to "google/gemma-3-270m"
        :type model_name: str, optional
        :param model_type: Type of model wrapper, defaults to "gemma"
        :type model_type: str, optional
        :param max_context_length: Maximum conversation context length, defaults to 10
        :type max_context_length: int, optional
        :return: None
        :rtype: None
        """
        self.model_name = model_name
        self.model_type = model_type
        self.max_context_length = max_context_length
        self.conversation_manager = None
        self.scene_graph_formatter = SceneGraphFormatter()
        
        # Initialize session state keys
        self._init_session_state()
    
    def _init_session_state(self) -> None:
        """Initialize Streamlit session state for chat interface.
        
        :return: None
        :rtype: None
        """
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        if "chat_intro_started" not in st.session_state:
            st.session_state.chat_intro_started = False
        
        if "conversation_manager" not in st.session_state:
            st.session_state.conversation_manager = None
        
        if "scene_graph_context" not in st.session_state:
            st.session_state.scene_graph_context = None
    
    def initialize_conversation_manager(self) -> None:
        """Initialize the conversation manager if not already done.
        
        :return: None
        :rtype: None
        """
        if st.session_state.conversation_manager is None:
            try:
                st.session_state.conversation_manager = create_conversation_manager(
                    model_name=self.model_name,
                    model_type=self.model_type,
                    max_context_length=self.max_context_length
                )
                logger.info("Conversation manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize conversation manager: {e}")
                st.error(f"Failed to initialize chat system: {e}")
    
    def set_scene_graph_context(self, scene_graph_data: Dict[str, Any]) -> None:
        """Set the current scene graph context for conversation.
        
        :param scene_graph_data: Scene graph detection results
        :type scene_graph_data: Dict[str, Any]
        :return: None
        :rtype: None
        """
        st.session_state.scene_graph_context = scene_graph_data
        
        if st.session_state.conversation_manager:
            st.session_state.conversation_manager.set_scene_graph(scene_graph_data)
            logger.info("Scene graph context updated")
    
    def get_chat_response(self, user_input: str) -> str:
        """Get a response to user input with scene graph context.
        
        :param user_input: User's message
        :type user_input: str
        :return: Generated response
        :rtype: str
        """
        if not st.session_state.conversation_manager:
            self.initialize_conversation_manager()
        
        if not st.session_state.conversation_manager:
            return "I apologize, but the chat system is not available. Please try again later."
        
        try:
            # Set scene graph context if available
            if st.session_state.scene_graph_context:
                st.session_state.conversation_manager.set_scene_graph(
                    st.session_state.scene_graph_context
                )
            
            # Get response
            response = st.session_state.conversation_manager.get_response(user_input)
            return response
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def render_chat_interface(self) -> None:
        """Render the complete chat interface in Streamlit.
        
        :return: None
        :rtype: None
        """
        # Initialize conversation manager if needed
        self.initialize_conversation_manager()
        
        # Show intro message if first time
        if not st.session_state.get("chat_intro_started", False):
            self._show_intro_message()
            st.session_state.chat_intro_started = True
        
        # Render chat messages
        self._render_chat_messages()
        
        # Render chat input
        self._render_chat_input()
        
        # Render control buttons
        self._render_control_buttons()
    
    def _show_intro_message(self) -> None:
        """Show introductory message for the chat interface.
        
        :return: None
        :rtype: None
        """
        intro_messages = [
            """Hello there! Welcome to VidSgg... 
            I'm your personal AI assistant for video scene graph analysis. 
            I can help you discover hidden relationships and objects in your videos!
            Just upload a video above to start.""",
        ]
        
        st.session_state.chat_messages = [
            {"message": intro_messages[0], "is_user": False}
        ]
        
        # Display intro message
        intro_container = st.empty()
        with intro_container.container():
            message(
                st.session_state.chat_messages[0]["message"],
                is_user=False,
                key=f"intro_0_{uuid.uuid4().hex[:8]}",
                allow_html=True,
            )
        intro_container.empty()
    
    def _render_chat_messages(self) -> None:
        """Render all chat messages in the interface.
        
        :return: None
        :rtype: None
        """
        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state.chat_messages):
                message(
                    msg["message"],
                    is_user=msg["is_user"],
                    key=f"chat_msg_{i}",
                    allow_html=True,
                )
    
    def _render_chat_input(self) -> None:
        """Render the chat input field.
        
        :return: None
        :rtype: None
        """
        def handle_chat_input():
            user_input = st.session_state.chat_input
            if user_input.strip():
                # Add user message
                st.session_state.chat_messages.append(
                    {"message": user_input, "is_user": True}
                )
                
                # Generate bot response
                bot_response = self.get_chat_response(user_input)
                st.session_state.chat_messages.append(
                    {"message": bot_response, "is_user": False}
                )
                
                # Clear input
                st.session_state.chat_input = ""
        
        st.text_input(
            "Ask me about your scene graph analysis:",
            key="chat_input",
            on_change=handle_chat_input,
            placeholder="Type your question here...",
        )
    
    def _render_control_buttons(self) -> None:
        """Render control buttons for the chat interface.
        
        :return: None
        :rtype: None
        """
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Clear Chat", help="Clear all chat messages"):
                st.session_state.chat_messages = []
                if st.session_state.conversation_manager:
                    st.session_state.conversation_manager.clear_history()
                st.rerun()
        
        with col2:
            if st.button("Show Context", help="Show current scene graph context"):
                self._show_scene_graph_context()
        
        with col3:
            if st.button("Export Chat", help="Export chat history"):
                self._export_chat_history()
    
    def _show_scene_graph_context(self) -> None:
        """Show the current scene graph context.
        
        :return: None
        :rtype: None
        """
        if st.session_state.scene_graph_context:
            with st.expander("Current Scene Graph Context", expanded=True):
                context_text = self.scene_graph_formatter.format_scene_graph(
                    st.session_state.scene_graph_context
                )
                st.text_area("Scene Description", context_text, height=200, disabled=True)
        else:
            st.info("No scene graph context available. Upload a video to generate scene graphs.")
    
    def _export_chat_history(self) -> None:
        """Export chat history to downloadable format.
        
        :return: None
        :rtype: None
        """
        if not st.session_state.chat_messages:
            st.warning("No chat history to export")
            return
        
        # Create export data
        export_data = {
            "model_info": {
                "model_name": self.model_name,
                "model_type": self.model_type,
            },
            "scene_graph_context": st.session_state.scene_graph_context,
            "chat_history": st.session_state.chat_messages,
            "export_timestamp": str(uuid.uuid4())
        }
        
        # Convert to JSON
        import json
        json_data = json.dumps(export_data, indent=2)
        
        # Create download button
        st.download_button(
            label="Download Chat History",
            data=json_data,
            file_name=f"scene_graph_chat_{uuid.uuid4().hex[:8]}.json",
            mime="application/json",
        )
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation.
        
        :return: Dictionary with conversation statistics
        :rtype: Dict[str, Any]
        """
        if not st.session_state.chat_messages:
            return {"total_messages": 0, "user_messages": 0, "bot_messages": 0}
        
        total_messages = len(st.session_state.chat_messages)
        user_messages = sum(1 for msg in st.session_state.chat_messages if msg["is_user"])
        bot_messages = total_messages - user_messages
        
        return {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "bot_messages": bot_messages,
            "has_scene_graph_context": st.session_state.scene_graph_context is not None
        }
    
    def add_system_message(self, message: str) -> None:
        """Add a system message to the chat.
        
        :param message: System message content
        :type message: str
        :return: None
        :rtype: None
        """
        st.session_state.chat_messages.append({
            "message": f"[System] {message}",
            "is_user": False
        })
    
    def clear_chat(self) -> None:
        """Clear all chat messages and reset conversation.
        
        :return: None
        :rtype: None
        """
        st.session_state.chat_messages = []
        if st.session_state.conversation_manager:
            st.session_state.conversation_manager.clear_history()
        st.session_state.chat_intro_started = False
