import json
import streamlit as st
from itertools import zip_longest
import difflib
import html
import re
from collections import defaultdict
from typing import Dict, Set, List, Union, Tuple, Optional
from llmvm.client.client import llm
from llmvm.common.objects import Message, TextContent, User
from models import ComparisonSet, ComparisonPair, ModelOutput

class ComparisonUI:
    def __init__(self, comparison_set: ComparisonSet, a_name: str, b_name: str, index: int):
        self.comparison_set = comparison_set
        self.a_name = a_name
        self.b_name = b_name
        self.index = index

    def is_code(self, text: Union[str, List[str]]) -> bool:
        """Check if text looks like code."""
        if not isinstance(text, str):
            if isinstance(text, list):
                text = ' '.join(text)
            else:
                return False

        indicators = ["def ", "class ", "import ", "<helpers>", "function", "var ", "let ", "const ", "python_version",
                      "aiofiles", "anyio", "asyncio", "```python", "```json",
                      "<script", "public ", "private ", "</helpers>", "<helpers_result>", "</helpers_result>", "answer("]
        return any(ind in text for ind in indicators)

    def html_diff(self, text1: str, text2: str) -> Tuple[str, Dict]:
        """Create HTML diff between two code blocks."""
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()

        # Generate diff and convert to HTML
        diff = difflib.ndiff(lines1, lines2)
        html_lines = []

        lines_added = 0
        lines_removed = 0

        for line in diff:
            if line.startswith('+ '):   # Added line
                lines_added += 1
                line = html.escape(line[2:])
                html_lines.append(f'<div class="diff-add">{line}</div>')
            elif line.startswith('- '): # Removed line
                lines_removed += 1
                line = html.escape(line[2:])
                html_lines.append(f'<div class="diff-remove">{line}</div>')
            elif line.startswith('  '): # Unchanged line
                line = html.escape(line[2:])
                html_lines.append(f'<div>{line}</div>')

        metrics = {
            'lines_added': lines_added,
            'lines_removed': lines_removed,
            'total_diff': lines_added + lines_removed
        }

        return ''.join(html_lines), metrics

    def calculate_text_metrics(self, text: Union[str, List[str]]) -> Dict:
        """Calculate metrics for text content."""
        if not isinstance(text, str):
            if isinstance(text, list):
                # For list outputs, join them with spaces
                text = " ".join(text)
            else:
                return {"word_count": 0, "unique_words": set(), "list_items": 0}

        # Count words
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        unique_words = set(words)

        # Count list items
        list_items = 0
        for line in text.split('\n'):
            if re.match(r'^\s*[-*•]|^\s*\d+\.', line.strip()):
                list_items += 1

        # Calculate readability (simplified Flesch Reading Ease)
        sentences = len(re.split(r'[.!?]+', text))
        if sentences == 0:
            sentences = 1  # Avoid division by zero
        avg_words_per_sentence = word_count / sentences

        # Detect code blocks
        is_code_block = self.is_code(text)

        return {
            "word_count": word_count,
            "unique_words": unique_words,
            "list_items": list_items,
            "avg_words_per_sentence": avg_words_per_sentence,
            "is_code_block": is_code_block
        }

    def expandable_code(self, key: str, code: str, char_limit: int = 600):
        """Display code with Show More/Less option if it exceeds char_limit."""
        if key not in st.session_state.expanded_texts:
            st.session_state.expanded_texts[key] = False  # Default collapsed state

        if len(code) > char_limit:
            if st.session_state.expanded_texts[key]:
                st.code(code, language="python")
                # Use custom button styling for right-aligned button
                st.markdown('<div class="show-more-button">', unsafe_allow_html=True)
                if st.button(f"Show Less", key=f"less_{key}"):
                    st.session_state.expanded_texts[key] = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Show truncated code with proper syntax highlighting
                truncated_code = code[:char_limit] + "..."
                st.code(truncated_code, language="python")
                # Use custom button styling for right-aligned button
                st.markdown('<div class="show-more-button">', unsafe_allow_html=True)
                if st.button(f"Show More", key=f"more_{key}"):
                    st.session_state.expanded_texts[key] = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.code(code, language="python")

    def expandable_text(self, key: str, text: str, char_limit: int = 600, is_metrics: bool = False):
        """Display text with Show More/Less option if it exceeds char_limit."""
        if key not in st.session_state.expanded_texts:
            st.session_state.expanded_texts[key] = False  # Default collapsed state

        if len(text) > char_limit:
            if st.session_state.expanded_texts[key]:
                if is_metrics:
                    st.markdown(text, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="model-text">{text}</div>', unsafe_allow_html=True)
                # Use custom button styling for right-aligned button
                st.markdown('<div class="show-more-button">', unsafe_allow_html=True)
                if st.button(f"Show Less", key=f"less_{key}"):
                    st.session_state.expanded_texts[key] = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                if is_metrics:
                    st.markdown(f"{text[:char_limit]}...", unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="model-text">{text[:char_limit]}...</div>', unsafe_allow_html=True)
                # Use custom button styling for right-aligned button
                st.markdown('<div class="show-more-button">', unsafe_allow_html=True)
                if st.button(f"Show More", key=f"more_{key}"):
                    st.session_state.expanded_texts[key] = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            if is_metrics:
                st.markdown(text, unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="model-text">{text}</div>', unsafe_allow_html=True)

    def toggle_expand_all(self):
        """Toggle expand/collapse all state."""
        st.session_state.expand_all = not st.session_state.expand_all

    def render_control_buttons(self):
        """Render control buttons for expand/collapse and summary."""
        col1, col2 = st.columns([1, 1])
        with col1:
            expand_label = "Collapse All" if st.session_state.expand_all else "Expand All"
            st.button(expand_label, on_click=self.toggle_expand_all, key=f"{self.index}_expand_all")

    def render_summary(self):
        """Render overall summary section."""

        overall_metrics = {
            'word_count_a': 0,
            'word_count_b': 0,
            'code_blocks_a': 0,
            'code_blocks_b': 0,
            'list_items_a': 0,
            'list_items_b': 0,
            'unique_words_a': set(),
            'unique_words_b': set(),
        }

        # Process all outputs to gather statistics
        for pair in self.comparison_set.pairs:
            metrics_a = self.calculate_text_metrics(pair.output_a.content)
            metrics_b = self.calculate_text_metrics(pair.output_b.content)

            # Update overall metrics
            overall_metrics['word_count_a'] += metrics_a['word_count']
            overall_metrics['word_count_b'] += metrics_b['word_count']
            overall_metrics['list_items_a'] += metrics_a['list_items']
            overall_metrics['list_items_b'] += metrics_b['list_items']

            if metrics_a['is_code_block']:
                overall_metrics['code_blocks_a'] += 1
            if metrics_b['is_code_block']:
                overall_metrics['code_blocks_b'] += 1

            overall_metrics['unique_words_a'].update(metrics_a['unique_words'])
            overall_metrics['unique_words_b'].update(metrics_b['unique_words'])

        # Display summary
        st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)

        # Use a more compact format with regular text
        word_diff = overall_metrics['word_count_b'] - overall_metrics['word_count_a']
        diff_sign = "+" if word_diff > 0 else ""
        st.write(f"**Total Words**: A: {overall_metrics['word_count_a']} | B: {overall_metrics['word_count_b']} ({diff_sign}{word_diff})")

        unique_diff = len(overall_metrics['unique_words_b']) - len(overall_metrics['unique_words_a'])
        diff_sign = "+" if unique_diff > 0 else ""
        st.write(f"**Unique Words**: A: {len(overall_metrics['unique_words_a'])} | B: {len(overall_metrics['unique_words_b'])} ({diff_sign}{unique_diff})")

        code_diff = overall_metrics['code_blocks_b'] - overall_metrics['code_blocks_a']
        diff_sign = "+" if code_diff > 0 else ""
        st.write(f"**Code Blocks**: A: {overall_metrics['code_blocks_a']} | B: {overall_metrics['code_blocks_b']} ({diff_sign}{code_diff})")

        # Calculate vocabulary overlap
        common_words = overall_metrics['unique_words_a'].intersection(overall_metrics['unique_words_b'])
        all_words = overall_metrics['unique_words_a'].union(overall_metrics['unique_words_b'])
        if all_words:
            overlap_percentage = len(common_words) / len(all_words) * 100
            st.write(f"Vocabulary overlap: {overlap_percentage:.1f}%")

        st.markdown("</div>", unsafe_allow_html=True)

    def render_metrics_html(self, metrics_a: Dict, metrics_b: Dict) -> str:
        """Generate HTML for metrics comparison."""
        metrics_html = '<div class="metrics-container">'

        # Word count comparison
        word_diff = metrics_b["word_count"] - metrics_a["word_count"]
        diff_sign = "+" if word_diff > 0 else ""
        metrics_html += f'<p><strong>Words</strong>: A: {metrics_a["word_count"]} | B: {metrics_b["word_count"]} ({diff_sign}{word_diff})</p>'

        # List items if any exist
        if metrics_a["list_items"] > 0 or metrics_b["list_items"] > 0:
            list_diff = metrics_b["list_items"] - metrics_a["list_items"]
            diff_sign = "+" if list_diff > 0 else ""
            metrics_html += f'<p><strong>List Items</strong>: A: {metrics_a["list_items"]} | B: {metrics_b["list_items"]} ({diff_sign}{list_diff})</p>'

        # Unique terms
        unique_in_a = metrics_a["unique_words"] - metrics_b["unique_words"]
        unique_in_b = metrics_b["unique_words"] - metrics_a["unique_words"]

        if unique_in_a or unique_in_b:
            diff = len(unique_in_b) - len(unique_in_a)
            diff_sign = "+" if diff > 0 else ""
            metrics_html += f'<p><strong>Unique Terms</strong>: A: {len(unique_in_a)} | B: {len(unique_in_b)} ({diff_sign}{diff})</p>'

            # Only show details if there are significant unique terms
            if len(unique_in_a) > 2 or len(unique_in_b) > 2:
                metrics_html += '<hr style="margin: 5px 0; border-color: #ddd">'
                if unique_in_a:
                    metrics_html += f'<p><strong>Only in A:</strong> {", ".join(list(unique_in_a)[:5])}</p>'
                if unique_in_b:
                    metrics_html += f'<p><strong>Only in B:</strong> {", ".join(list(unique_in_b)[:5])}</p>'

        metrics_html += '</div>'
        return metrics_html

    def render_model_names(self):
        col1, diff_col, col2 = st.columns([2, 3, 2])
        with col1:
            st.header(f"Model A: {self.a_name}")
        with col2:
            st.header(f"Model B: {self.b_name}")

    def render_comparison_pair(self, pair: ComparisonPair, index: int):
        """Render a single comparison pair."""
        output_a = pair.output_a
        output_b = pair.output_b
        commentary = pair.commentary

        st.markdown(f"### Comparison {index + 1}")

        # Calculate metrics
        metrics_a = self.calculate_text_metrics(output_a.content)
        metrics_b = self.calculate_text_metrics(output_b.content)

        # Convert lists to strings if needed
        a_content = output_a.content if isinstance(output_a.content, str) else "\n".join(output_a.content)
        b_content = output_b.content if isinstance(output_b.content, str) else "\n".join(output_b.content)

        # Determine if both outputs are code for diff view
        both_code = self.is_code(a_content) and self.is_code(b_content)

        if both_code:
            col1, diff_col, col2 = st.columns([2, 3, 2])
        else:
            col1, metrics_col, col2 = st.columns([2, 3, 2])

        # Model A Output
        with col1:
            if len(output_a.content) > 1:
                with st.expander(output_a.role.capitalize(), expanded=st.session_state.expand_all):
                    if self.is_code(a_content):
                        self.expandable_code(f"{self.index}_code_a_{index}", a_content)
                    else:
                        st.markdown(f'<div class="model-text">{a_content}</div>', unsafe_allow_html=True)

        # Diff View or Metrics
        if both_code:
            with diff_col:  # type: ignore
                with st.expander("Diff View", expanded=st.session_state.expand_all):
                    diff_html, diff_metrics = self.html_diff(a_content, b_content)

                    if len(diff_html) > 500:
                        self.expandable_text(f"{self.index}_diff_{index}", diff_html, char_limit=500, is_metrics=True)
                    else:
                        st.markdown(f'<div>{diff_html}</div>', unsafe_allow_html=True)

                    # Display diff metrics in compact format
                    diff_metrics_html = '<div class="metrics-container">'
                    diff_metrics_html += f'<p><strong>Lines Changed</strong>: {diff_metrics["total_diff"]} (+{diff_metrics["lines_added"]} / -{diff_metrics["lines_removed"]})</p>'
                    diff_metrics_html += '</div>'
                    st.markdown(diff_metrics_html, unsafe_allow_html=True)

                if commentary and len(commentary) > 0:
                    with st.expander("Commentary", expanded=st.session_state.expand_all):
                        if commentary and len(commentary) > 150:
                            self.expandable_text(f"{self.index}_commentary_{index}", commentary, char_limit=150, is_metrics=True)
                        else:
                            st.markdown(f'<div>{commentary}</div>', unsafe_allow_html=True)
        else:
            # Metrics in an expander
            metrics_html = self.render_metrics_html(metrics_a, metrics_b)
            with metrics_col:  # type: ignore
                with st.expander("Metrics", expanded=st.session_state.expand_all):
                    if len(metrics_html) > 300:
                        self.expandable_text(f"{self.index}_metrics_{index}", metrics_html, char_limit=300, is_metrics=True)
                    else:
                        st.markdown(metrics_html, unsafe_allow_html=True)

                if commentary and len(commentary) > 0:
                    with st.expander("Commentary", expanded=st.session_state.expand_all):
                        if commentary and len(commentary) > 150:
                            self.expandable_text(f"{self.index}_commentary_{index}", commentary, char_limit=150, is_metrics=True)
                        else:
                            st.markdown(f'<div>{commentary}</div>', unsafe_allow_html=True)

        # Model B Output
        with col2:
            if len(output_b.content) > 1:
                with st.expander(output_b.role.capitalize(), expanded=st.session_state.expand_all):
                    if isinstance(output_b.content, list):  # Multiple text boxes inside the expander
                        for text in output_b.content:
                            st.markdown(f'<div class="model-text">{text}</div>', unsafe_allow_html=True)
                    elif self.is_code(b_content):  # Code detection
                        self.expandable_code(f"{self.index}_code_b_{index}", b_content)
                    else:
                        self.expandable_text(f"{self.index}_text_{index}", b_content)  # Handles long text dynamically

        # Add an arrow between sections
        st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

    def render(self):
        """Render the complete comparison UI."""
        if 'show_metrics' not in st.session_state:
            st.session_state.show_metrics = True
            # Add a toggle for showing/hiding metrics
            st.session_state.show_metrics = st.checkbox("Show Metrics", value=st.session_state.show_metrics)

        self.render_control_buttons()

        if st.session_state.show_metrics:
            self.render_summary()

        self.render_model_names()

        # Add a filter for message types
        message_types = ['user', 'assistant']
        selected_types = st.multiselect("Filter by message type", message_types, default=message_types, key=f"{self.index}_message_types")

        for idx, pair in enumerate(self.comparison_set.pairs):
            if pair.output_a.role in selected_types:
                self.render_comparison_pair(pair, idx)


class ComparisonUIContainer:
    def __init__(self):
        self.comparison_uis: List[ComparisonUI] = []

        self.init_session_state()
        self.setup_css()

    def init_session_state(self):
        """Initialize session state variables."""
        if 'expand_all' not in st.session_state:
            st.session_state.expand_all = True
        if 'expanded_texts' not in st.session_state:
            st.session_state.expanded_texts = {}

    def setup_css(self):
        """Set up CSS styles for the UI."""
        st.markdown("""
        <style>
        .block-container {
            max-width: 90%;
        }
        .left-border {
            border-left: 5px solid;
            padding: 10px;
            margin: 10px 0;
        }
        .model-a {
            border-left-color: green;
            background-color: #f9f9f9;
        }
        .model-b {
            border-left-color: red;
            background-color: #f0f0f0;
        }
        .model-text {
            border-left-color: gray;
            background-color: #f9f9f9;
            padding: 10px;
            margin: 10px 0;
            overflow-wrap: break-word;
        }
        .arrow {
            text-align: center;
            font-size: 24px;
            color: gray;
            margin: -10px 0;
        }
        .diff-add {
            background-color: #d6ffd6;
            text-decoration: none;
        }
        .diff-remove {
            background-color: #ffdddd;
            text-decoration: none;
        }
        .summary-card {
            border-bottom: 1px solid #eee;
            padding: 10px 0 15px 0;
            margin-bottom: 20px;
        }
        .metrics-container {
            font-size: 0.75rem;
            margin-top: 8px;
            margin-bottom: 8px;
            color: #666;
        }
        .metrics-container .st-emotion-cache-16txtl3 h1 {
            font-size: 0.8rem !important;
            font-weight: normal !important;
        }
        .metrics-container .st-emotion-cache-1xarl3l {
            font-size: 0.75rem !important;
        }
        .metrics-container .st-emotion-cache-1offfbd p {
            font-size: 0.75rem !important;
            margin-bottom: 0 !important;
        }
        .metrics-value {
            font-size: 0.75rem !important;
        }
        .st-emotion-cache-1l1k1dn .st-emotion-cache-q8sbsg p {
            font-size: 0.75rem !important;
        }
        .metrics-container .st-emotion-cache-50yulf {
            margin-top: 0 !important;
            margin-bottom: 4px !important;
        }
        .metrics-container .st-emotion-cache-1iyw5u {
            display: none;  /* Hide the placeholder gap */
        }
        .stMarkdown:has(h3) {
            display: none;  /* Hide comparison headers */
        }
        /* Custom button styling */
        .show-more-button {
            display: flex;
            justify-content: flex-end;
            margin-top: 2px;
        }
        .show-more-button button {
            height: 1.2rem !important;
            padding: 0rem 0.8rem !important;
            font-size: 0.7rem !important;
            line-height: 1 !important;
            min-height: unset !important;
            border-radius: 3px !important;
        }
        /* Override default Streamlit button styles */
        .show-more-button .st-emotion-cache-1xarl3l {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        .stButton > button {
            border: 1px solid #ddd !important;
        }
        </style>
        """, unsafe_allow_html=True)

    def add_comparison_ui(self, comparison_set: ComparisonSet, a_name: str, b_name: str):
        comparison_ui = ComparisonUI(comparison_set=comparison_set, a_name=a_name, b_name=b_name, index=len(self.comparison_uis))
        self.comparison_uis.append(comparison_ui)

    def render(self):
        for comparison_ui in self.comparison_uis:
            comparison_ui.render()

