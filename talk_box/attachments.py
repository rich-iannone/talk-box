import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import chatlas

@dataclass
class AttachmentMetadata:
    """
    Metadata for individual file attachments.

    This class captures essential information about file processing results, enabling debugging,
    performance monitoring, and analytics for file attachment workflows.

    The metadata is automatically collected during file processing and can be accessed via the
    `Attachments.metadata` property for inspection and logging.

    Parameters
    ----------
    filename
        The name of the file (without path).
    file_type
        File extension without the dot (e.g., 'pdf', 'png', 'py').
    size_bytes
        File size in bytes.
    content_type
        Category of content: 'image', 'pdf', 'text', 'error', 'unsupported'.
    processing_time_ms
        Time taken to process the file in milliseconds.
    error
        Error message if processing failed.

    Examples
    --------
    **Accessing File Metadata**

    ```python
    import talk_box as tb

    files = tb.Attachments("report.pdf", "image.png", "data.csv")

    # Process files (happens automatically during chat)
    bot = tb.ChatBot().provider_model("openai:gpt-4-turbo")
    conversation = bot.chat(files.with_prompt("Analyze these files"))

    # Inspect metadata
    for meta in files.metadata:
        print(f"File: {meta.filename}")
        print(f"Type: {meta.content_type}")
        print(f"Size: {meta.size_bytes:,} bytes")
        print(f"Processing time: {meta.processing_time_ms:.1f}ms")
        if meta.error:
            print(f"Error: {meta.error}")
        print("---")
    ```

    **Performance Monitoring**

    ```python
    # Monitor processing performance for optimization
    large_files = tb.Attachments("big_report.pdf", "large_image.png")

    # ... process files ...

    total_time = sum(m.processing_time_ms for m in large_files.metadata)
    total_size = sum(m.size_bytes for m in large_files.metadata)

    print(f"Processed {len(large_files.metadata)} files")
    print(f"Total size: {total_size:,} bytes")
    print(f"Total time: {total_time:.1f}ms")
    print(f"Avg speed: {total_size/total_time*1000:.0f} bytes/sec")
    ```

    **Error Detection and Handling**

    ```python
    files = tb.Attachments("good_file.pdf", "missing_file.txt", "corrupted.png")

    # ... process files ...

    # Check for errors
    failed_files = [m for m in files.metadata if m.error]
    successful_files = [m for m in files.metadata if not m.error]

    print(f"Successfully processed: {len(successful_files)} files")
    if failed_files:
        print("Failed files:")
        for meta in failed_files:
            print(f"  {meta.filename}: {meta.error}")
    ```
    """

    filename: str
    file_type: str
    size_bytes: int
    content_type: str
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

    def __post_init__(self):
        """Validate metadata after initialization."""
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if self.processing_time_ms is not None and self.processing_time_ms < 0:
            raise ValueError("processing_time_ms must be non-negative")


class Attachments:
    """
    File attachment handler for Talk Box conversations.

    The Attachments class enables you to include files in your AI conversations for analysis,
    review, and discussion. It automatically handles different file types (text, images, PDFs) and
    integrates seamlessly with `ChatBot` for programmatic conversations.

    **Primary Use Cases:**

    - **Code Review**: attach source files for automated code analysis
    - **Document Analysis**: process PDFs, reports, and documentation
    - **Data Analysis**: include CSV, JSON, or other data files for insights
    - **Content Generation**: attach references for context-aware content creation
    - **Image Analysis**: process diagrams, charts, or photos with vision models
    - **Research Assistance**: attach papers, articles, or research materials

    **Key Features:**

    - multi-file support with automatic content type detection
    - rich metadata collection for debugging and analytics
    - seamless ChatBot integration for programmatic workflows
    - chainable API following Talk Box design patterns
    - error handling with graceful fallbacks

    Parameters
    ----------
    *file_paths
        Variable number of file paths to attach to the conversation.

    Examples
    --------
    **Single File Analysis**

    ```python
    import talk_box as tb

    # Analyze a single document
    files = tb.Attachments("quarterly_report.pdf").with_prompt(
        "Summarize the key financial metrics and trends in this report."
    )

    bot = tb.ChatBot().provider_model("openai:gpt-4-turbo")
    analysis = bot.chat(files)
    ```

    **Code Review Workflow**

    ```python
    # Review multiple source files
    code_files = tb.Attachments(
        "src/main.py",
        "src/utils.py",
        "tests/test_main.py"
    ).with_prompt(
        "Review this Python code for bugs, performance issues, and best practices. "
        "Focus on the main logic and test coverage."
    )

    reviewer = (
        tb.ChatBot()
        .provider_model("openai:gpt-4-turbo")
        .preset("technical_advisor")
        .temperature(0.3)
    )

    review = reviewer.chat(code_files)
    ```

    **Data Analysis Pipeline**

    ```python
    # Analyze data files with context
    data_analysis = tb.Attachments(
        "sales_data.csv",
        "customer_segments.json",
        "analysis_notes.md"
    ).with_prompt(
        "Analyze the sales trends, identify top customer segments, "
        "and suggest actionable insights based on the data and notes provided."
    )

    analyst = (
        tb.ChatBot()
        .provider_model("openai:gpt-4-turbo")
        .temperature(0.4)
        .max_tokens(2000)
    )

    insights = analyst.chat(data_analysis)
    ```

    **Image and Document Combination**

    ```python
    # Combine visual and textual content
    presentation_review = (
        tb.Attachments(
            "slide_deck.pdf",
            "speaker_notes.md",
            "chart_image.png"
        ).with_prompt(
            "Review this presentation for clarity, visual impact, and alignment "
            "between slides and speaker notes. Suggest improvements."
        )
    )

    presentation_bot = tb.ChatBot().provider_model("openai:gpt-4-turbo")
    feedback = presentation_bot.chat(presentation_review)
    ```

    **Batch Processing Multiple Files**

    ```python
    # Process multiple documents for comparison
    for file_path in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
        analysis = tb.Attachments(file_path).with_prompt(
            "Extract the main thesis and key arguments from this document."
        )
        result = bot.chat(analysis)
        print(f"Analysis of {file_path}:")
        print(result)
    ```

    **Jupyter Notebook Integration**

    ```python
    # HTML display in Jupyter notebooks
    files = tb.Attachments("code.py", "data.csv", "report.pdf").with_prompt(
        "Analyze these project files for insights and recommendations."
    )

    # Just displaying the object shows an HTML summary
    files  # Displays file count, sizes, types, and prompt in formatted HTML
    ```

    ```python
    # Then process with ChatBot
    bot = tb.ChatBot().provider_model("openai:gpt-4-turbo")
    result = bot.chat(files)
    ```

    The result here can also be displayed with HTML formatting:

    ```python
    result
    ```

    Notes
    -----
    - file attachments are designed for **single-turn programmatic conversations**
    - for interactive multi-turn conversations, use `bot.show("browser")` instead
    - large files are automatically chunked and processed efficiently
    - unsupported file types are handled gracefully with informative errors
    - all file processing includes timing and error metadata for debugging
    - **HTML representation**: displays rich summary in Jupyter notebooks so just print the object!
    """

    def __init__(self, *file_paths: Union[str, Path]):
        """Initialize with file paths to attach."""
        self.file_paths = [Path(p) for p in file_paths]
        self.metadata: List[AttachmentMetadata] = []
        self._contents: List[Any] = []
        self._prompt_text: str = ""
        self._processed = False

    def with_prompt(self, prompt: str) -> "Attachments":
        """
        Add a text prompt to accompany the file attachments.

        This method enables the fluent interface for combining prompt text with file attachments,
        following Talk Box's chainable API design. The prompt provides context and instructions
        for how the AI should analyze or interact with the attached files.

        Parameters
        ----------
        prompt
            The text prompt to include with the file attachments. This should provide clear
            instructions about what you want the AI to do with the attached files.

        Returns
        -------
        Attachments
            Returns self for method chaining.

        Examples
        --------
        **Specific Analysis Request**

        ```python
        import talk_box as tb

        files = (
            tb.Attachments("financial_report.pdf")
            .with_prompt(
                "Extract the key financial metrics and identify any concerning trends "
                "in this quarterly report. Focus on revenue, profit margins, and cash flow."
            )
        )
        ```

        **Code Review with Specific Criteria**

        ```python
        code_review = tb.Attachments("src/main.py", "tests/test_main.py").with_prompt(
            "Review this Python code for:\n"
            "1. Code quality and best practices\n"
            "2. Potential bugs or security issues\n"
            "3. Test coverage and completeness\n"
            "4. Performance optimization opportunities"
        )
        ```

        **Creative Content Generation**

        ```python
        references = (
            tb.Attachments("brand_guide.pdf", "competitor_analysis.md")
            .with_prompt(
                "Based on our brand guidelines and competitor analysis, create a "
                "marketing strategy for our new product launch. Focus on differentiation "
                "and brand consistency."
            )
        )
        ```

        **Data Analysis with Context**

        ```python
        data_files = (
            tb.Attachments("sales_data.csv", "market_context.md")
            .with_prompt(
                "Analyze the sales data in the context of the market information provided. "
                "Identify trends, anomalies, and actionable insights for the sales team."
            )
        )
        ```

        **Multi-file Comparison**

        ```python
        comparison = (
            tb.Attachments("version1.py", "version2.py")
            .with_prompt(
                "Compare these two versions of the code and explain:\n"
                "- What changed between versions\n"
                "- Whether the changes improve or degrade the code\n"
                "- Any potential issues introduced"
            )
        )
        ```

        Notes
        -----
        - the prompt is combined with file content when sent to the AI model
        - clear, specific prompts lead to better analysis results
        - you can include formatting instructions (bullets, sections, etc.)
        - the prompt applies to all attached files collectively
        """
        self._prompt_text = prompt
        return self

    @property
    def prompt(self) -> str:
        """Get the prompt text."""
        return self._prompt_text

    @property
    def files(self) -> List[Path]:
        """Get the list of file paths."""
        return self.file_paths

    def _process_files(self) -> List[Any]:
        """
        Process all files into content objects.

        This method handles the conversion of file paths into content objects, managing errors
        gracefully and collecting metadata for each file processed.

        Returns
        -------
        List
            List of content objects.
        """
        if self._processed:
            return self._contents

        self._contents = []
        self.metadata = []

        for file_path in self.file_paths:
            metadata, content = self._process_single_file(file_path)
            self.metadata.append(metadata)
            if content is not None:
                self._contents.append(content)

        self._processed = True
        return self._contents

    def _process_single_file(self, file_path: Path) -> tuple[AttachmentMetadata, Optional[Any]]:
        """
        Process a single file and return metadata + content.

        Parameters
        ----------
        file_path
            Path to the file to process.

        Returns
        -------
        tuple[AttachmentMetadata, Optional[Any]]
            Tuple of (metadata, content) where content is None if processing failed.
        """
        start_time = time.time()

        try:
            # Check file existence
            if not file_path.exists():
                metadata = AttachmentMetadata(
                    filename=file_path.name,
                    file_type=file_path.suffix[1:] if file_path.suffix else "unknown",
                    size_bytes=0,
                    content_type="error",
                    error=f"File not found: {file_path}",
                )
                return metadata, None

            # Get file info
            file_size = file_path.stat().st_size
            file_ext = file_path.suffix.lower()

            # Process based on file type
            content, content_type = self._process_by_type(file_path, file_ext)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            metadata = AttachmentMetadata(
                filename=file_path.name,
                file_type=file_ext[1:] if file_ext else "unknown",
                size_bytes=file_size,
                content_type=content_type,
                processing_time_ms=processing_time,
            )

            return metadata, content

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            metadata = AttachmentMetadata(
                filename=file_path.name,
                file_type=file_path.suffix[1:] if file_path.suffix else "unknown",
                size_bytes=file_path.stat().st_size if file_path.exists() else 0,
                content_type="error",
                processing_time_ms=processing_time,
                error=str(e),
            )
            return metadata, None

    def _process_by_type(self, file_path: Path, file_ext: str) -> tuple[Optional[Any], str]:
        """
        Process file based on its type using appropriate chatlas functions.

        Parameters
        ----------
        file_path
            Path to the file.
        file_ext
            File extension (lowercase, with dot).

        Returns
        -------
        tuple[Optional[Any], str]
            Tuple of (content_object, content_type_string).
        """
        # Image files - use chatlas image processing
        if file_ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]:
            try:
                content = chatlas.content_image_file(str(file_path), resize="high")
                return content, "image"
            except Exception as e:
                if "Pillow" in str(e):
                    # Provide helpful error for missing Pillow
                    raise ImportError(
                        "Image processing requires Pillow. Install with: pip install Pillow"
                    ) from e
                raise

        # PDF files - use chatlas PDF processing
        elif file_ext == ".pdf":
            content = chatlas.content_pdf_file(file_path)
            return content, "pdf"

        # Text-based files - process as formatted text
        elif file_ext in [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".json",
            ".csv",
            ".yaml",
            ".yml",
            ".xml",
            ".html",
        ]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()

                # Determine language for syntax highlighting
                lang_map = {
                    ".py": "python",
                    ".js": "javascript",
                    ".json": "json",
                    ".yaml": "yaml",
                    ".yml": "yaml",
                    ".xml": "xml",
                    ".html": "html",
                    ".md": "markdown",
                }
                lang = lang_map.get(file_ext, "")

                # Format as code block for better LLM processing
                content = f"File: {file_path.name}\n```{lang}\n{file_content}\n```"
                return content, "text"

            except UnicodeDecodeError:
                # Try with different encoding for problematic files
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        file_content = f.read()
                    content = (
                        f"File: {file_path.name} (decoded with latin-1)\n```\n{file_content}\n```"
                    )
                    return content, "text"
                except Exception:
                    # If all else fails, treat as binary
                    return None, "unsupported"
        else:
            # Unsupported file type
            return None, "unsupported"

    def to_chat_contents(self) -> List[Any]:
        """
        Convert attachments to chatlas-compatible content list.

        This method processes all attached files and converts them into the appropriate chatlas
        content objects. Images become ContentImageInline objects, PDFs become ContentPDF objects,
        and text files are combined into the prompt text.

        Returns
        -------
        List[Any]
            List containing text prompt and Content objects ready to pass
            directly to chatlas Chat.chat() method.

        Examples
        --------
        ```python
        import talk_box as tb

        files = tb.Attachments("image.png", "doc.pdf").with_prompt("Analyze these")
        contents = files.to_chat_contents()
        # Result: ["Analyze these", ContentImageInline(...), ContentPDF(...)]
        ```
        """
        # Ensure files are processed
        self._process_files()

        # Separate text content from media content
        text_content_parts = []
        media_contents = []

        # Process each file by examining both metadata and content
        for metadata in self.metadata:
            # Find the corresponding content in _contents
            # We need to process files again to get the content for each metadata
            file_path = next((fp for fp in self.file_paths if fp.name == metadata.filename), None)
            if file_path is None:
                continue

            # Get the content for this specific file
            _, content = self._process_single_file(file_path)

            if content is None:
                continue

            if metadata.content_type in ["text", "error"]:
                # For text files and errors, include in the prompt text
                if isinstance(content, str):
                    text_content_parts.append(content)
            else:
                # For images, PDFs, etc., keep as Content objects
                media_contents.append(content)

        # Build the final content list
        contents = []

        # Combine prompt text with text file contents
        combined_text_parts = []
        if self._prompt_text:
            combined_text_parts.append(self._prompt_text)

        if text_content_parts:
            combined_text_parts.extend(text_content_parts)

        # Handle mixed content (text + media) vs pure text
        if media_contents:
            # When we have media files, convert text to Content objects for consistency
            if combined_text_parts:
                combined_text = "\n\n".join(combined_text_parts)
                # Convert text to a Content object to match media content types
                from chatlas._content import ContentText

                text_content = ContentText(text=combined_text)
                contents.append(text_content)

            # Add all media content objects
            contents.extend(media_contents)
        else:
            # Pure text files - return as single string (existing behavior)
            if combined_text_parts:
                combined_text = "\n\n".join(combined_text_parts)
                contents.append(combined_text)

        return contents

    def get_metadata(self) -> List[AttachmentMetadata]:
        """
        Get metadata for all processed attachments.

        This method returns detailed metadata about file processing results, including timing
        information, file sizes, and any errors encountered. Useful for debugging, analytics, and
        user feedback.

        Returns
        -------
        List[AttachmentMetadata]
            List of metadata objects, one per file processed.
        """
        if not self._processed:
            self._process_files()  # Ensure files are processed
        return self.metadata.copy()

    def summary(self) -> str:
        """
        Get a human-readable summary of attached files.

        Creates a concise summary showing the number of files, total size, and breakdown by content
        type. Useful for logging, user interfaces, and debugging.

        Returns
        -------
        str
            Formatted summary string like "📎 2/3 files attached (1.2MB): 1 image, 1 pdf [1 failed]"

        Examples
        --------
        ```python
        import talk_box as tb

        files = tb.Attachments("code.py", "missing.txt", "diagram.png")
        print(files.summary())
        # Output: "📎 2/3 files attached (15.2KB): 1 text, 1 image [1 failed]"
        ```
        """
        if not self._processed:
            self._process_files()

        if not self.metadata:
            return "No files attached"

        total_files = len(self.metadata)
        successful = len([m for m in self.metadata if not m.error])
        failed = total_files - successful

        # Categorize by content type
        type_counts = {}
        total_size = 0

        for meta in self.metadata:
            if not meta.error:
                type_counts[meta.content_type] = type_counts.get(meta.content_type, 0) + 1
                total_size += meta.size_bytes

        # Format size
        if total_size < 1024:
            size_str = f"{total_size} bytes"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        else:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"

        # Build summary
        summary = f"📎 {successful}/{total_files} files attached ({size_str})"

        if type_counts:
            types_str = ", ".join([f"{count} {ftype}" for ftype, count in type_counts.items()])
            summary += f": {types_str}"

        if failed > 0:
            summary += f" [{failed} failed]"

        return summary

    def __len__(self) -> int:
        """Return number of file paths."""
        return len(self.file_paths)

    def __bool__(self) -> bool:
        """Return True if any files are attached."""
        return len(self.file_paths) > 0

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self._prompt_text:
            return f"Attachments({', '.join(str(p) for p in self.file_paths)}).with_prompt({self._prompt_text!r})"
        return f"Attachments({', '.join(str(p) for p in self.file_paths)})"

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        # Process files to get metadata
        if not self._processed:
            self._process_files()

        # File status summary
        total_files = len(self.metadata) if self.metadata else len(self.file_paths)
        successful = len([m for m in self.metadata if not m.error]) if self.metadata else 0
        failed = total_files - successful

        # Calculate total size
        total_size = sum(m.size_bytes for m in self.metadata if not m.error) if self.metadata else 0

        # Format size
        if total_size < 1024:
            size_str = f"{total_size} bytes"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        else:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"

        # Status color based on success rate
        if failed == 0:
            status_color = "#28a745"  # green
            status_icon = "✅"
        elif successful > 0:
            status_color = "#ffc107"  # yellow
            status_icon = "⚠️"
        else:
            status_color = "#dc3545"  # red
            status_icon = "❌"

        html = f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 8px 0; background: #f8f9fa; color: #212529;">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <div style="background: {status_color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-right: 10px;">
                    � Attachments
                </div>
                <div style="color: {status_color}; font-weight: bold;">
                    {status_icon} {successful}/{total_files} files ({size_str})
                </div>
            </div>
        """

        # File list
        if self.metadata:
            html += '<div style="margin-bottom: 12px; color: #212529;"><strong>Files:</strong></div><ul style="margin: 0; padding-left: 20px;">'
            for meta in self.metadata:
                if meta.error:
                    html += f'<li style="color: #dc3545;">❌ {meta.filename} - {meta.error}</li>'
                else:
                    # Format file size
                    if meta.size_bytes < 1024:
                        file_size = f"{meta.size_bytes} bytes"
                    elif meta.size_bytes < 1024 * 1024:
                        file_size = f"{meta.size_bytes / 1024:.1f} KB"
                    else:
                        file_size = f"{meta.size_bytes / (1024 * 1024):.1f} MB"

                    # Icon based on content type
                    type_icons = {
                        "text": "📄",
                        "image": "🖼️",
                        "pdf": "📕",
                        "error": "❌",
                        "unsupported": "❓",
                    }
                    icon = type_icons.get(meta.content_type, "📄")

                    html += f'<li>{icon} {meta.filename} <span style="color: #495057;">({file_size})</span></li>'
            html += "</ul>"
        else:
            # No metadata yet - show file paths
            html += '<div style="margin-bottom: 12px;"><strong>Files:</strong></div><ul style="margin: 0; padding-left: 20px;">'
            for file_path in self.file_paths:
                html += f"<li>📄 {file_path.name}</li>"
            html += "</ul>"

        # Prompt section
        if self._prompt_text:
            # Clean up prompt text - strip leading/trailing whitespace
            display_prompt = self._prompt_text.strip()

            html += f"""
            <div style="margin-top: 12px;">
                <strong style="color: #212529;">Prompt:</strong>
                <div style="background: #e9ecef; color: #212529; padding: 8px; border-radius: 4px; margin-top: 4px; font-family: monospace; font-size: 0.9em; max-height: 120px; overflow-y: auto; white-space: pre-wrap;">{display_prompt}</div>
            </div>
            """

        html += "</div>"
        return html
