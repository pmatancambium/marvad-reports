import streamlit as st
from pymongo import MongoClient
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from bson import ObjectId
import colorsys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from functools import lru_cache
from pymongo import ASCENDING, DESCENDING

# Add this at the beginning of your script, right after the imports
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# Add this right after the check_password function
if check_password():
    # Your existing app code goes here
    # Connect to MongoDB
    mongo_uri = st.secrets["MONGO_URI"]
    client = MongoClient(mongo_uri)
    db = client["marvad"]
    collection = db["plagiarism_reports"]
    collection_documents = db["documents"]

    # Function to get connected documents with similarity scores
    @lru_cache(maxsize=None)
    def get_connected_documents(doc_id):
        connected_reports = list(collection.find({"target_document": doc_id}))
        connected_docs = []
        for report in connected_reports:
            for result in report['results']:
                connected_docs.append({
                    'doc_id': result['document_id'],
                    'average_score': result['average_score']
                })
        return connected_docs

    # Function to get filename from document ID
    @lru_cache(maxsize=None)
    def get_filename(doc_id):
        return document_id_to_filename.get(str(doc_id), str(doc_id))

    # Function to calculate node size based on similarity score
    def get_node_size(score, min_size=1, max_size=10):
        return min_size + (max_size - min_size) * score

    def get_color(score):
        # Convert score to a color (red for low scores, green for high scores)
        hue = score * 0.3  # Hue value between 0 (red) and 0.3 (green)
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

    def get_edge_width(score, min_width=0.5, max_width=2):
        return min_width + (max_width - min_width) * score

    # Create indexes for frequently queried fields
    # collection.create_index([("target_document", ASCENDING)])
    # collection.create_index([("found_plagiarism", ASCENDING)])
    # collection.create_index([("created_at", DESCENDING)])
    # collection_documents.create_index([("_id", ASCENDING)])

    # Fetch data from MongoDB more efficiently
    reports = list(collection.find({}, {
        "_id": 1,
        "target_document": 1,
        "filename": 1,
        "found_plagiarism": 1,
        "total_sentences": 1,
        "total_paragraphs": 1,  # Add this line
        "total_images": 1,  # Add this line
        "results": 1,
        "created_at": 1,
        "plagiarized_document_ids": 1
    }))

    documents = list(collection_documents.find({}, {"_id": 1, "filename": 1, "name": 1}))

    # Create a mapping of document IDs to filenames
    document_id_to_filename = {str(doc["_id"]): doc.get("filename") or doc.get("name") or str(doc["_id"]) for doc in documents}

    # Convert ObjectId to string and add filename
    for report in reports:
        report['_id'] = str(report['_id'])
        report['filename'] = document_id_to_filename.get(report['target_document'], "Unknown")

    # Convert data to DataFrame
    df = pd.DataFrame(reports)

    # Optimize data processing
    document_stats = (
        pd.DataFrame(reports)
        .groupby('filename')
        .agg({
            'found_plagiarism': 'sum',
            'total_sentences': 'max'
        })
        .reset_index()
        .sort_values(['found_plagiarism', 'total_sentences'], ascending=[False, False])
    )

    # Sidebar for filtering
    st.sidebar.header("Filter Options")
    target_filename = st.sidebar.selectbox(
        "Select Target Document", 
        ["All"] + list(document_stats['filename']),
        format_func=lambda x: f"{x} {'(Plagiarized)' if x != 'All' and document_stats[document_stats['filename'] == x]['found_plagiarism'].values[0] > 0 else ''}"
    )
    found_plagiarism = st.sidebar.selectbox("Found Plagiarism", ["All", True, False])

    # Optimize filtering
    @st.cache_data
    def filter_data(df, target_filename, found_plagiarism):
        filtered = df
        if target_filename != "All":
            filtered = filtered[filtered["filename"] == target_filename]
        if found_plagiarism != "All":
            filtered = filtered[filtered["found_plagiarism"] == (found_plagiarism == "True")]
        return filtered

    filtered_df = filter_data(df, target_filename, found_plagiarism)

    # Main layout
    st.title("Plagiarism Report Dashboard")

    # Create tabs
    tab1, tab2 = st.tabs(["Overall Dashboard", "Document Specific"])

    with tab1:
        # Overview section
        st.header("Overview")

        # Calculate additional metrics
        total_reports = len(df)
        plagiarism_found = df["found_plagiarism"].sum()
        unique_documents = df["target_document"].nunique()
        avg_similarity_score = df["results"].apply(lambda x: x[0]["average_score"] if x else 0).mean()
        recent_reports = df[df["created_at"] > (datetime.now() - timedelta(days=7))].shape[0]

        # Create two rows of metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Reports", total_reports)
        col2.metric("Plagiarism Found", plagiarism_found)
        # col3.metric("Unique Documents", unique_documents)

        col4, col5 = st.columns(2)
        col4.metric("Avg. Similarity Score", f"{avg_similarity_score:.2f}")
        # col5.metric("Recent Reports (7 days)", recent_reports)
        col5.metric("Filtered Reports", len(filtered_df))

        # Plagiarism Distribution
        st.subheader("Plagiarism Distribution")
        fig_dist = px.histogram(df, x="found_plagiarism", color="found_plagiarism",
                                 labels={"found_plagiarism": "Plagiarism Found", "count": "Number of Reports"},
                                 title="Distribution of Plagiarism Findings")
        st.plotly_chart(fig_dist)

        # Top Plagiarized Documents
        # st.subheader("Top Plagiarized Documents")
        # top_plagiarized = df[df["found_plagiarism"] == True].groupby("filename").size().sort_values(ascending=False).head(10)
        # fig_top = px.bar(top_plagiarized, x=top_plagiarized.index, y=top_plagiarized.values,
        #                  labels={"x": "Document", "y": "Number of Plagiarism Instances"},
        #                  title="Top 10 Most Plagiarized Documents")
        # fig_top.update_xaxes(tickangle=45)
        # st.plotly_chart(fig_top)

        # Plagiarism Severity Distribution
        st.subheader("Plagiarism Severity Distribution")
        severity_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        severity_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        df['severity'] = pd.cut(df['results'].apply(lambda x: x[0]['average_score'] if x else 0), 
                                bins=severity_bins, labels=severity_labels)
        fig_severity = px.pie(df, names='severity', title="Distribution of Plagiarism Severity")
        st.plotly_chart(fig_severity)

        # # Plagiarism Detection Timeline
        # st.subheader("Plagiarism Detection Timeline")
        # df['created_at'] = pd.to_datetime(df['created_at'])
        # timeline_data = df.set_index('created_at').resample('D')['found_plagiarism'].sum().reset_index()
        # fig_timeline = px.line(timeline_data, x='created_at', y='found_plagiarism',
        #                        labels={'created_at': 'Date', 'found_plagiarism': 'Plagiarism Instances'},
        #                        title='Plagiarism Detection Timeline')
        # st.plotly_chart(fig_timeline)

        # Top 10 Documents with Highest Plagiarism Scores
        st.subheader("Top 10 Documents with Highest Plagiarism Scores")

        def safe_max_score(x):
            try:
                scores = [r.get('average_score', 0) for r in x.iloc[0] if isinstance(r, dict)]
                return max(scores) if scores else 0
            except Exception:
                return 0

        top_plagiarism_scores = df.groupby('filename')['results'].apply(safe_max_score).nlargest(10)

        if not top_plagiarism_scores.empty:
            fig_top_scores = px.bar(top_plagiarism_scores, x=top_plagiarism_scores.index, y=top_plagiarism_scores.values,
                                    labels={'x': 'Document', 'y': 'Highest Plagiarism Score'},
                                    title='Top 10 Documents with Highest Plagiarism Scores')
            fig_top_scores.update_xaxes(tickangle=45)
            st.plotly_chart(fig_top_scores)
        else:
            st.write("No plagiarism scores available for documents.")

        # st.subheader("Plagiarism Detection Accuracy")

        # if 'confidence_label' in df.columns:
        #     accuracy_data = df.groupby('confidence_label')['found_plagiarism'].mean().reset_index()
        #     fig_accuracy = px.bar(accuracy_data, x='confidence_label', y='found_plagiarism',
        #                           labels={'confidence_label': 'Confidence Label', 'found_plagiarism': 'Plagiarism Detection Rate'},
        #                           title='Plagiarism Detection Accuracy by Confidence Label')
        #     st.plotly_chart(fig_accuracy)
        # else:
        #     st.write("Confidence label information is not available in the dataset.")
        
        #     # Alternative visualization: Overall plagiarism detection rate
        #     overall_accuracy = df['found_plagiarism'].mean()
        #     fig_overall = px.bar(x=['Overall'], y=[overall_accuracy],
        #                          labels={'x': 'Category', 'y': 'Plagiarism Detection Rate'},
        #                          title='Overall Plagiarism Detection Rate')
        #     st.plotly_chart(fig_overall)

        # Plagiarism Type Distribution
        st.subheader("Plagiarism Type Distribution")
        plagiarism_types = df[df['found_plagiarism'] == True].apply(lambda row: [
            'Sentences' if row['results'][0]['num_matches_sentences'] > 0 else None,
            'Paragraphs' if row['results'][0]['num_matches_paragraphs'] > 0 else None,
            'Images' if row['results'][0]['num_matches_images'] > 0 else None
        ], axis=1).explode().dropna().value_counts()

        fig_types = px.pie(values=plagiarism_types.values, names=plagiarism_types.index,
                           title='Distribution of Plagiarism Types')
        st.plotly_chart(fig_types)

        # Plagiarism Score Distribution
        st.subheader("Plagiarism Score Distribution")

        def safe_max_score(results):
            try:
                scores = [r.get('average_score', 0) for r in results if isinstance(r, dict)]
                return max(scores) if scores else 0
            except Exception:
                return 0

        df['max_plagiarism_score'] = df['results'].apply(safe_max_score)

        fig_score_dist = px.histogram(df, x='max_plagiarism_score',
                                      labels={'max_plagiarism_score': 'Max Plagiarism Score'},
                                      title='Overall Plagiarism Score Distribution')
        st.plotly_chart(fig_score_dist)

        # If you still want to show the distribution by confidence label (if available):
        if 'confidence_label' in df.columns:
            fig_score_box = px.box(df, x='confidence_label', y='max_plagiarism_score',
                                   labels={'confidence_label': 'Confidence Label', 'max_plagiarism_score': 'Max Plagiarism Score'},
                                   title='Plagiarism Score Distribution by Confidence Label')
            st.plotly_chart(fig_score_box)

        # # Interactive Similarity Score Distribution
        # st.subheader("Similarity Score Distribution")
        # similarity_scores = df["results"].apply(lambda x: x[0]["average_score"] if x else 0)
        # fig_similarity = px.histogram(similarity_scores, nbins=20,
        #                               labels={"value": "Similarity Score", "count": "Number of Reports"},
        #                               title="Distribution of Similarity Scores")
        # fig_similarity.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
        # st.plotly_chart(fig_similarity)


    with tab2:
        if target_filename != "All":
            st.header(f"Analysis for: {target_filename}")
            
            # Create subtabs for Document Connection Graph and Detailed Results
            subtab1, subtab2 = st.tabs(["Document Connection Graph", "Detailed Results"])
            
            with subtab1:
                st.subheader("Document Connection Graph")
                
                # Filtering options
                min_similarity = st.slider("Minimum Similarity Score", 0.0, 1.0, 0.0, 0.01)
                max_connections = st.number_input("Max Connections per Node", 1, 100, 10)
                
                # Create a Pyvis network with adjusted initial zoom
                net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white")
                net.set_options("""
                var options = {
                    "nodes": {
                        "font": {
                            "size": 12
                        }
                    },
                    "edges": {
                        "color": {
                            "inherit": true
                        },
                        "smooth": false
                    },
                    "physics": {
                        "enabled": true,
                        "barnesHut": {
                            "gravitationalConstant": -2000,
                            "centralGravity": 0.3,
                            "springLength": 95
                        },
                        "minVelocity": 0.75
                    },
                    "interaction": {
                        "zoomView": true
                    }
                }
                """)

                # Find the document ID for the target filename
                target_doc_id = next((str(doc["_id"]) for doc in documents if doc.get("filename") == target_filename or doc.get("name") == target_filename), None)
                
                if target_doc_id:
                    net.add_node(target_filename, size=10, color='#ff0000', title=target_filename, label=target_filename)  # Red for the main document
                    
                    # First level connections
                    first_level = get_connected_documents(target_doc_id)
                    for doc in sorted(first_level, key=lambda x: x['average_score'], reverse=True)[:max_connections]:
                        if doc['average_score'] >= min_similarity:
                            filename = get_filename(doc['doc_id'])
                            node_size = get_node_size(doc['average_score'])
                            node_color = get_color(doc['average_score'])
                            net.add_node(filename, size=node_size, color=node_color, title=f"{filename}\nSimilarity: {doc['average_score']:.2f}", label=filename)
                            net.add_edge(target_filename, filename, width=get_edge_width(doc['average_score']))
                            
                            # Second level connections
                            second_level = get_connected_documents(doc['doc_id'])
                            for second_doc in sorted(second_level, key=lambda x: x['average_score'], reverse=True)[:max_connections]:
                                if second_doc['average_score'] >= min_similarity:
                                    second_filename = get_filename(second_doc['doc_id'])
                                    second_node_size = get_node_size(second_doc['average_score'])
                                    second_node_color = get_color(second_doc['average_score'])
                                    net.add_node(second_filename, size=second_node_size, color=second_node_color, title=f"{second_filename}\nSimilarity: {second_doc['average_score']:.2f}", label=second_filename)
                                    net.add_edge(filename, second_filename, width=get_edge_width(second_doc['average_score']))
                
                # Save and display the graph
                net.save_graph("graph.html")
                HtmlFile = open("graph.html", 'r', encoding='utf-8')
                source_code = HtmlFile.read()
                components.html(source_code, height=800)

                # Display graph statistics
                st.write(f"Total nodes in graph: {len(net.nodes)}")
                st.write(f"Total edges in graph: {len(net.edges)}")

            with subtab2:
                st.subheader("Detailed Results")
                if not filtered_df.empty:
                    for index, row in filtered_df.iterrows():
                        plagiarism_indicator = "🚫 " if row['found_plagiarism'] else ""
                        st.subheader(f"{plagiarism_indicator}Report ID: {row['_id']} - {row['filename']}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Found Plagiarism", str(row['found_plagiarism']))
                        col2.metric("Total Sentences", row['total_sentences'])
                        col3.metric("Total Paragraphs", row.get('total_paragraphs', 'N/A'))
                        col4.metric("Total Images", row.get('total_images', 'N/A'))

                        st.write("**Similar Documents Found:**")
                        similar_docs = [document_id_to_filename.get(doc_id, doc_id) for doc_id in row['plagiarized_document_ids']]
                        st.write(", ".join(similar_docs))
                        
                        for result in row["results"]:
                            with st.expander(f"Similar Document: {document_id_to_filename.get(result['document_id'], result['document_id'])}"):
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Average Score", f"{result['average_score']:.2f}")
                                col2.metric("Confidence", f"{result['confidence_percentage']}%")
                                col3.metric("Confidence Label", result['confidence_label'])
                                
                                st.write("**Matches:**")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Sentences", f"{result['num_matches_sentences']} ({result['percentage_sentences_matched']}%)")
                                col2.metric("Paragraphs", f"{result['num_matches_paragraphs']} ({result['percentage_paragraphs_matched']}%)")
                                col3.metric("Images", f"{result['num_matches_images']} ({result['percentage_images_matched']}%)")
                                
                                st.write("**Top Matches (Sentences):**")
                                for match in result['top_matches_sentences']:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("Target Sentence:")
                                        st.info(match['target_sentence'])
                                    with col2:
                                        st.write("Matching Sentence:")
                                        st.warning(match['matching_sentence'])
                                    if 'score' in match:
                                        st.write(f"Similarity Score: {match['score']:.2f}")
                                    st.write("---")
                                
                                st.write("**Top Matches (Paragraphs):**")
                                for match in result['top_matches_paragraphs']:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("Target Paragraph:")
                                        st.info(match['target_paragraph'])
                                    with col2:
                                        st.write("Matching Paragraph:")
                                        st.warning(match['matching_paragraph'])
                                    if 'score' in match:
                                        st.write(f"Similarity Score: {match['score']:.2f}")
                                    st.write("---")
                                
                                st.write("**Top Matches (Images):**")
                                for match in result['top_matches_images']:
                                    st.write(f"- {match}")
                        
                        st.write("---")
        else:
            st.info("Please select a specific document from the sidebar to view its analysis.")

    # Add a download button for the filtered data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="plagiarism_reports.csv",
        mime="text/csv",
    )

else:
    st.stop()  # Don't run the rest of the app.
