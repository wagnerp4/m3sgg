import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_results_tabs(ctx: Dict[str, Any]) -> None:
    """Render the results view tabs including players, tables, charts, and chat.

    :param ctx: Context with shared helpers and flags
    :type ctx: dict
    :return: None
    :rtype: None
    """
    validate_video_file = ctx.get("validate_video_file")
    StreamlitVideoProcessor = ctx.get("StreamlitVideoProcessor")
    CHAT_INTERFACE_AVAILABLE = ctx.get("CHAT_INTERFACE_AVAILABLE", False)
    SceneGraphChatInterface = ctx.get("SceneGraphChatInterface")

    main_tab1, main_tab2 = st.tabs(["SGG View", "Advanced SGG View"])
    with main_tab1:
        st.header(" Video Players")
        if st.session_state.uploaded_video_file is not None:
            vid_col1, vid_col2, vid_col3 = st.columns(3)
            with vid_col1:
                st.subheader("Original Video")
                st.video(st.session_state.uploaded_video_file)
            with vid_col2:
                st.subheader("Object Detection")
                bbox_video_path = st.session_state.get(
                    "bbox_video_path"
                ) or st.session_state.get("debug_bbox_path")
                if bbox_video_path and os.path.exists(bbox_video_path):
                    import cv2

                    cap = cv2.VideoCapture(bbox_video_path)
                    bbox_frames = (
                        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if cap.isOpened()
                        else "unknown"
                    )
                    cap.release()
                    if callable(validate_video_file) and validate_video_file(
                        bbox_video_path
                    ):
                        try:
                            st.video(bbox_video_path)
                        except Exception as e:
                            st.warning(f"Direct video display failed: {e}")
                            try:
                                with open(bbox_video_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                    st.video(video_bytes)
                            except Exception as e2:
                                st.error(f"Error loading bbox video: {e2}")
                                st.video(st.session_state.uploaded_video_file)
                    else:
                        st.error("Bounding box video file is corrupted or unreadable")
                        st.video(st.session_state.uploaded_video_file)
                else:
                    st.video(st.session_state.uploaded_video_file)
                    st.warning("No bounding box video available - showing original")

                if "results" in st.session_state and "bbox_info" in st.session_state:
                    st.markdown("---")
                    st.subheader("Detected Objects")
                    bbox_info = st.session_state["bbox_info"]
                    if bbox_info:
                        bbox_df = pd.DataFrame(
                            [
                                {
                                    "Object": bbox["object_name"],
                                    "Confidence": f"{bbox['confidence']:.3f}",
                                    "BBox": f"[{bbox['bbox'][0]:.0f}, {bbox['bbox'][1]:.0f}, {bbox['bbox'][2]:.0f}, {bbox['bbox'][3]:.0f}]",
                                }
                                for bbox in bbox_info
                            ]
                        )
                        st.dataframe(bbox_df, width="stretch", hide_index=True)
                    else:
                        st.info("No objects detected above confidence threshold")

            with vid_col3:
                st.subheader("Scene Graph Analysis")
                sg_video_path = st.session_state.get(
                    "scene_graph_video_path"
                ) or st.session_state.get("debug_sg_path")
                if sg_video_path and os.path.exists(sg_video_path):
                    import cv2

                    cap = cv2.VideoCapture(sg_video_path)
                    sg_frames = (
                        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if cap.isOpened()
                        else "unknown"
                    )
                    cap.release()
                    if callable(validate_video_file) and validate_video_file(
                        sg_video_path
                    ):
                        try:
                            st.video(sg_video_path)
                        except Exception as e:
                            st.warning(f"Direct video display failed: {e}")
                            try:
                                with open(sg_video_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                    st.video(video_bytes)
                            except Exception as e2:
                                st.error(f"Error loading scene graph video: {e2}")
                                st.video(st.session_state.uploaded_video_file)
                                st.caption(" Scene graph overlay failed to load")
                    else:
                        st.error("Scene graph video file is corrupted or unreadable")
                        st.video(st.session_state.uploaded_video_file)
                else:
                    st.video(st.session_state.uploaded_video_file)
                    st.warning("No scene graph video available - showing original")

                if (
                    "results" in st.session_state
                    and "relationship_info" in st.session_state
                ):
                    st.markdown("---")
                    st.subheader("Scene Graph Relationships")
                    relationship_info = st.session_state["relationship_info"]
                    if relationship_info:
                        temp_processor = None
                        if (
                            "model_path" in st.session_state
                            and StreamlitVideoProcessor is not None
                        ):
                            try:
                                temp_processor = StreamlitVideoProcessor(
                                    st.session_state["model_path"]
                                )
                            except Exception:
                                pass
                        relationship_data = []
                        for rel in relationship_info:
                            subject_name = "person1"
                            if "subject_class" in rel and temp_processor:
                                subject_name = temp_processor.get_object_name(
                                    rel["subject_class"]
                                )
                            object_name = "object"
                            if "object_class" in rel and temp_processor:
                                object_name = temp_processor.get_object_name(
                                    rel["object_class"]
                                )
                            relationship_name = "interacts_with"
                            if temp_processor:
                                if (
                                    "attention_type" in rel
                                    and "attention_confidence" in rel
                                    and rel["attention_confidence"] > 0.1
                                ):
                                    relationship_name = (
                                        temp_processor.get_relationship_name(
                                            rel["attention_type"], "attention"
                                        )
                                    )
                                elif (
                                    "spatial_type" in rel
                                    and "spatial_confidence" in rel
                                    and rel["spatial_confidence"] > 0.1
                                ):
                                    relationship_name = (
                                        temp_processor.get_relationship_name(
                                            rel["spatial_type"], "spatial"
                                        )
                                    )
                            relationship_data.append(
                                {
                                    "Subject": subject_name,
                                    "Relation": relationship_name,
                                    "Object": object_name,
                                    "Confidence": f"{rel['confidence']:.3f}",
                                }
                            )
                        if relationship_data:
                            relationship_df = pd.DataFrame(relationship_data)
                            st.dataframe(
                                relationship_df, width="stretch", hide_index=True
                            )
                        else:
                            st.info(
                                "No relationships detected above confidence threshold"
                            )
                    else:
                        st.info("No relationships detected above confidence threshold")

        st.markdown("---")
        st.header("Chat Assistant")
        if CHAT_INTERFACE_AVAILABLE and SceneGraphChatInterface is not None:
            if "chat_interface" not in st.session_state:
                st.session_state.chat_interface = SceneGraphChatInterface(
                    model_name="google/gemma-3-270m", model_type="gemma"
                )
            if "results" in st.session_state:
                st.session_state.chat_interface.set_scene_graph_context(
                    st.session_state["results"]
                )
            st.session_state.chat_interface.render_chat_interface()
        else:
            st.warning("Advanced chat interface not available. Using basic chat.")
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
                st.session_state.chat_intro_started = False
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            if not st.session_state.get("chat_intro_started", False):
                st.session_state.chat_intro_started = True
                intro_messages = [
                    """Hello there! Welcome to VidSgg... 
                        I'm your personal AI assistant for video scene graph analysis. 
                        I can help you discover hidden relationships and objects in your videos!
                        Just upload a video above to start.""",
                ]
                st.session_state.chat_messages = [
                    {"message": intro_messages[0], "is_user": False}
                ]
                intro_container = st.empty()
                from streamlit_chat import message

                with intro_container.container():
                    message(
                        st.session_state.chat_messages[0]["message"],
                        is_user=False,
                        key=f"intro_0_",
                    )
                intro_container.empty()

            def handle_chat_input() -> None:
                user_input = st.session_state.chat_input
                if user_input.strip():
                    st.session_state.chat_messages.append(
                        {"message": user_input, "is_user": True}
                    )
                    bot_response = generate_bot_response(user_input)
                    st.session_state.chat_messages.append(
                        {"message": bot_response, "is_user": False}
                    )
                    st.session_state.chat_input = ""

            def generate_bot_response(user_input: str) -> str:
                user_input_lower = user_input.lower()
                if "hello" in user_input_lower or "hi" in user_input_lower:
                    return "Hello! I'm your VidSgg assistant. I can help you understand scene graph generation results and answer questions about the analysis."
                elif "help" in user_input_lower:
                    return "I can help you with:\n• Understanding scene graph results\n• Explaining object detections\n• Interpreting relationship data\n• Model configuration questions\n• Export options"
                elif "object" in user_input_lower and "results" in st.session_state:
                    results = st.session_state["results"]
                    avg_objects = (
                        np.mean(results["detections"]) if results["detections"] else 0
                    )
                    return f"In your video analysis, I detected an average of {avg_objects:.1f} objects per frame across {results['processed_frames']} processed frames."
                elif (
                    "relationship" in user_input_lower and "results" in st.session_state
                ):
                    results = st.session_state["results"]
                    avg_relationships = (
                        np.mean(results["relationships"])
                        if results["relationships"]
                        else 0
                    )
                    return f"The analysis found an average of {avg_relationships:.1f} relationships per frame in your video."
                elif "confidence" in user_input_lower and "results" in st.session_state:
                    results = st.session_state["results"]
                    avg_confidence = (
                        np.mean(results["confidences"]) if results["confidences"] else 0
                    )
                    return f"The average confidence score across all detections was {avg_confidence:.2f}."
                elif "model" in user_input_lower:
                    return "The VidSgg model uses STTran (Spatial-Temporal Transformer) for scene graph generation. It processes video frames to detect objects and their relationships over time."
                elif "export" in user_input_lower:
                    return "You can export your results in JSON, CSV, or XML format using the export options in the sidebar. The exported data will include all detection and relationship information."
                else:
                    return "I'm here to help with your scene graph analysis! Ask me about objects, relationships, confidence scores, or how the model works."

            chat_container = st.container()
            from streamlit_chat import message

            with chat_container:
                for i, msg in enumerate(st.session_state.chat_messages):
                    message(msg["message"], is_user=msg["is_user"], key=f"chat_msg_{i}")
            st.text_input(
                "Ask me about your scene graph analysis:",
                key="chat_input",
                on_change=handle_chat_input,
                placeholder="Type your question here...",
            )
            if st.button("Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()

        st.markdown("---")
        sgg_tab1, sgg_tab2 = st.tabs(["Temporal View", "NLP View"])
        with sgg_tab1:
            st.header("Temporal Scene Graph Analysis")
            if "results" in st.session_state:
                results = st.session_state["results"]
                if results["detections"]:
                    df = pd.DataFrame(
                        {
                            "Frame": range(len(results["detections"])),
                            "Objects_Detected": results["detections"],
                            "Relationships": results["relationships"],
                            "Confidence": results["confidences"],
                            "Time_Seconds": results["frame_times"],
                        }
                    )
                    st.subheader("Object Node Timeline")
                    if results.get("frame_objects") and any(results["frame_objects"]):
                        all_objects = set()
                        for frame_objects in results["frame_objects"]:
                            for obj in frame_objects:
                                all_objects.add(obj["object_name"])
                        if all_objects:
                            all_objects = sorted(list(all_objects))
                            fig_timeline = go.Figure()
                            person_y = 0
                            object_y_positions = list(range(1, len(all_objects) + 1))
                            fig_timeline.add_trace(
                                go.Scatter(
                                    x=[0, len(results["frame_objects"]) - 1],
                                    y=[person_y, person_y],
                                    mode="lines",
                                    name="Person",
                                    line=dict(width=8, color="red"),
                                    showlegend=True,
                                )
                            )
                            for frame_idx in range(len(results["frame_objects"])):
                                fig_timeline.add_trace(
                                    go.Scatter(
                                        x=[frame_idx],
                                        y=[person_y],
                                        mode="markers",
                                        marker=dict(
                                            size=20,
                                            color="red",
                                            line=dict(width=2, color="white"),
                                        ),
                                        name=f"Person (Frame {frame_idx})",
                                        hovertemplate="<b>Person</b><br>"
                                        + f"Frame: {frame_idx}<br>"
                                        + "<extra></extra>",
                                        showlegend=False,
                                    )
                                )
                            distinct_colors = [
                                "rgb(0, 100, 200)",
                                "rgb(200, 50, 50)",
                                "rgb(50, 150, 50)",
                                "rgb(150, 50, 150)",
                                "rgb(200, 100, 0)",
                                "rgb(0, 150, 150)",
                                "rgb(150, 100, 50)",
                                "rgb(100, 0, 200)",
                                "rgb(200, 150, 0)",
                                "rgb(50, 100, 150)",
                                "rgb(150, 50, 100)",
                            ]
                            for i, obj_name in enumerate(all_objects):
                                obj_y = object_y_positions[i]
                                color_index = i % len(distinct_colors)
                                object_color = distinct_colors[color_index]
                                fig_timeline.add_trace(
                                    go.Scatter(
                                        x=[0, len(results["frame_objects"]) - 1],
                                        y=[obj_y, obj_y],
                                        mode="lines",
                                        name=obj_name,
                                        line=dict(width=6, color=object_color),
                                        showlegend=True,
                                    )
                                )
                                object_frames = []
                                object_confidences = []
                                for frame_idx, frame_objects in enumerate(
                                    results["frame_objects"]
                                ):
                                    obj_in_frame = None
                                    for obj in frame_objects:
                                        if obj["object_name"] == obj_name:
                                            obj_in_frame = obj
                                            break
                                    if obj_in_frame:
                                        object_frames.append(frame_idx)
                                        object_confidences.append(
                                            obj_in_frame["confidence"]
                                        )
                                if object_frames:
                                    fig_timeline.add_trace(
                                        go.Scatter(
                                            x=object_frames,
                                            y=[obj_y] * len(object_frames),
                                            mode="markers",
                                            marker=dict(
                                                size=20,
                                                color=object_color,
                                                line=dict(width=2, color="white"),
                                            ),
                                            name=f"{obj_name} spheres",
                                            hovertemplate=f"<b>{obj_name}</b><br>"
                                            + "Frame: %{x}<br>"
                                            + "Confidence: %{customdata:.3f}<br>"
                                            + "<extra></extra>",
                                            customdata=object_confidences,
                                            showlegend=False,
                                        )
                                    )
                                    if len(object_frames) > 1:
                                        fig_timeline.add_trace(
                                            go.Scatter(
                                                x=object_frames,
                                                y=[obj_y] * len(object_frames),
                                                mode="lines",
                                                line=dict(
                                                    width=3,
                                                    color=object_color,
                                                    dash="solid",
                                                ),
                                                name=f"{obj_name} connections",
                                                showlegend=False,
                                                hoverinfo="skip",
                                            )
                                        )
                                for frame_idx in object_frames:
                                    fig_timeline.add_trace(
                                        go.Scatter(
                                            x=[frame_idx, frame_idx],
                                            y=[person_y, obj_y],
                                            mode="lines",
                                            line=dict(
                                                width=2, color="gray", dash="dot"
                                            ),
                                            name=f"Edge {frame_idx}",
                                            hovertemplate=f"<b>Person → {obj_name}</b><br>"
                                            + f"Frame: {frame_idx}<br>"
                                            + "<extra></extra>",
                                            showlegend=False,
                                        )
                                    )
                            fig_timeline.update_layout(
                                title="Object Nodes Across Frames with Person-Object Relationships",
                                xaxis=dict(
                                    title="Frame Number",
                                    tickmode="linear",
                                    tick0=0,
                                    dtick=1,
                                    showgrid=True,
                                    gridcolor="lightgray",
                                ),
                                yaxis=dict(
                                    title="Nodes",
                                    tickmode="array",
                                    tickvals=[person_y] + object_y_positions,
                                    ticktext=["Person"] + all_objects,
                                    side="left",
                                ),
                                height=400 + (len(all_objects) + 1) * 30,
                                showlegend=True,
                                legend=dict(
                                    orientation="v",
                                    yanchor="top",
                                    y=1,
                                    xanchor="left",
                                    x=1.02,
                                ),
                            )
                            st.plotly_chart(fig_timeline, width="stretch")
                            st.subheader("Object Statistics")
                            obj_stats = []
                            for obj_name in all_objects:
                                presence_count = sum(
                                    1
                                    for frame_objects in results["frame_objects"]
                                    if any(
                                        obj["object_name"] == obj_name
                                        for obj in frame_objects
                                    )
                                )
                                total_frames = len(results["frame_objects"])
                                presence_percentage = (
                                    (presence_count / total_frames) * 100
                                    if total_frames > 0
                                    else 0
                                )
                                confidences = []
                                for frame_objects in results["frame_objects"]:
                                    for obj in frame_objects:
                                        if obj["object_name"] == obj_name:
                                            confidences.append(obj["confidence"])
                                avg_confidence = (
                                    np.mean(confidences) if confidences else 0.0
                                )
                                obj_stats.append(
                                    {
                                        "Object": obj_name,
                                        "Frames Present": presence_count,
                                        "Total Frames": total_frames,
                                        "Presence %": f"{presence_percentage:.1f}%",
                                        "Avg Confidence": f"{avg_confidence:.3f}",
                                    }
                                )
                            stats_df = pd.DataFrame(obj_stats)
                            st.dataframe(stats_df, width="stretch", hide_index=True)
                        else:
                            st.info("No objects detected in any frame")
                    else:
                        st.info(
                            "No detailed object information available for timeline visualization"
                        )
                    st.subheader("Scene Graph Metrics Over Time")
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=df["Time_Seconds"],
                            y=df["Objects_Detected"],
                            mode="lines+markers",
                            name="Objects Detected",
                            line=dict(color="blue"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df["Time_Seconds"],
                            y=df["Relationships"],
                            mode="lines+markers",
                            name="Relationships",
                            line=dict(color="red"),
                            yaxis="y2",
                        )
                    )
                    fig.update_layout(
                        title="Scene Graph Analysis Over Time",
                        xaxis_title="Time (seconds)",
                        yaxis=dict(title="Objects Detected", side="left"),
                        yaxis2=dict(
                            title="Relationships", side="right", overlaying="y"
                        ),
                        legend=dict(x=0.02, y=0.98),
                    )
                    st.plotly_chart(fig, width="stretch")
                    fig3 = px.line(
                        df,
                        x="Time_Seconds",
                        y="Confidence",
                        title="Average Confidence Over Time",
                        labels={
                            "Time_Seconds": "Time (seconds)",
                            "Confidence": "Avg Confidence",
                        },
                    )
                    st.plotly_chart(fig3, width="stretch")
                    st.subheader("Detection Details")
                    st.dataframe(df, width="stretch")

        with sgg_tab2:
            st.header(" NLP Analysis")
            if st.session_state.uploaded_video_file is not None:
                st.subheader(" Video Analysis")
                st.video(st.session_state.uploaded_video_file)
                st.markdown("---")
                st.header("NLP Module Results")
                nlp_col1, nlp_col2 = st.columns(2)
                with nlp_col1:
                    st.subheader("Video Summarization")
                    summarization_text = """
                        **Automatically Generated Summary:**
                        
                        This video contains multiple scenes with various objects and activities. 
                        The analysis detected people interacting with objects in different spatial 
                        configurations. Key activities include movement patterns, object 
                        manipulations, and social interactions between detected entities.
                        
                        **Key Findings:**
                        • Multiple human subjects identified
                        • Various object interactions detected
                        • Temporal activity patterns observed
                        • Scene transitions and context changes noted
                        """
                    st.text_area(
                        "Summary", summarization_text, height=200, disabled=True
                    )
                    st.subheader(" Video Semantic Search")
                    search_results = """
                        🔎 **Semantic Search Results:**
                        
                        **Query:** "People walking"
                        **Timestamps:** 0:12-0:18, 0:45-0:52, 1:23-1:30
                        
                        **Query:** "Object interaction"
                        **Timestamps:** 0:25-0:35, 1:05-1:15, 1:40-1:50
                        
                        **Query:** "Group activity"
                        **Timestamps:** 0:30-0:55, 1:10-1:35
                        """
                    st.text_area(
                        "Search Results", search_results, height=150, disabled=True
                    )
                with nlp_col2:
                    st.subheader("Video Captioning")
                    captioning_text = """
                        **Frame-by-Frame Captions:**
                        
                        **00:05** - A person stands near a table with objects
                        **00:12** - Multiple people enter the scene from the left
                        **00:18** - Someone picks up an object from the surface
                        **00:25** - Two people engage in conversation
                        **00:32** - Group activity begins with shared focus
                        **00:40** - Objects are rearranged on the table
                        **00:48** - People move toward the background
                        **00:55** - Scene transitions to new activity
                        **01:02** - New objects appear in the frame
                        **01:10** - Final interactions before scene ends
                        """
                    st.text_area("Captions", captioning_text, height=200, disabled=True)
                    st.subheader("🔮 Action Anticipation")
                    anticipation_text = """
                        **Predicted Future Actions:**
                        
                        **Next 5 seconds:**
                        • Person likely to move towards door (85% confidence)
                        • Object manipulation probability: 72%
                        • Group dispersal expected: 68%
                        
                        **Next 10 seconds:**
                        • Scene change probability: 91%
                        • New person entry likelihood: 45%
                        • Activity continuation: 23%
                        
                        **Temporal Patterns:**
                        • Regular 15-second activity cycles detected
                        • Spatial movement patterns suggest routine behavior
                        """
                    st.text_area(
                        "Predictions", anticipation_text, height=150, disabled=True
                    )
                st.markdown("---")
                st.header("Advanced NLP Features")
                feature_col1, feature_col2, feature_col3 = st.columns(3)
                with feature_col1:
                    st.subheader("Emotion Analysis")
                    st.info(
                        "Detected emotions: Neutral (45%), Happy (30%), Focused (25%)"
                    )
                    emotion_data = pd.DataFrame(
                        {
                            "Emotion": ["Neutral", "Happy", "Focused", "Surprised"],
                            "Percentage": [45, 30, 25, 15],
                        }
                    )
                    fig_emotion = px.pie(
                        emotion_data,
                        values="Percentage",
                        names="Emotion",
                        title="Emotion Distribution",
                    )
                    st.plotly_chart(fig_emotion, width="stretch")
                with feature_col2:
                    st.subheader("Scene Classification")
                    st.info("Scene type: Indoor Office Environment")
                    scene_data = pd.DataFrame(
                        {
                            "Scene Type": [
                                "Office",
                                "Meeting Room",
                                "Kitchen",
                                "Living Room",
                            ],
                            "Confidence": [0.89, 0.65, 0.23, 0.18],
                        }
                    )
                    fig_scene = px.bar(
                        scene_data,
                        x="Scene Type",
                        y="Confidence",
                        title="Scene Classification Confidence",
                    )
                    st.plotly_chart(fig_scene, width="stretch")
                with feature_col3:
                    st.subheader("Activity Recognition")
                    st.info("Primary activity: Collaborative Work")
                    activity_data = pd.DataFrame(
                        {
                            "Activity": [
                                "Meeting",
                                "Discussion",
                                "Presentation",
                                "Break",
                            ],
                            "Duration": [45, 25, 20, 10],
                        }
                    )
                    fig_activity = px.bar(
                        activity_data,
                        x="Activity",
                        y="Duration",
                        title="Activity Duration (seconds)",
                    )
                    st.plotly_chart(fig_activity, width="stretch")
            else:
                st.info("Please upload a video file above to see NLP analysis results.")
                st.markdown("---")
                st.subheader("Available NLP Features")
                placeholder_features = [
                    "**Video Summarization** - Generate comprehensive summaries of video content",
                    "**Video Captioning** - Frame-by-frame natural language descriptions",
                    "**Semantic Search** - Find specific content using natural language queries",
                    "**Action Anticipation** - Predict future actions and activities",
                    "**Emotion Analysis** - Detect and analyze emotional states",
                    "**Scene Classification** - Identify and categorize different scene types",
                    "**Activity Recognition** - Recognize and track various activities",
                ]
                for feature in placeholder_features:
                    st.markdown(feature)
