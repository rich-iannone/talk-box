import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

try:
    import chatlas
except ImportError:
    raise ImportError(
        "chatlas is required for file attachment support. Install it with: pip install chatlas"
    )


@dataclass
class AttachmentMetadata:
    """
    Metadata for individual file attachments.

    This class captures essential information about file processing results, including performance
    metrics and error details for debugging and analytics.

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

    Provides an interface for adding files to chat messages. The class handles multiple file types,
    provides rich metadata for debugging and analytics, and converts files into content objects for
    LLM integration.

    Parameters
    ----------
    *file_paths
        Variable number of file paths to attach to the conversation.

    Examples
    --------
    ### Basic file attachment

    ```python
    from talk_box import Attachments

    # Single file
    files = Attachments("report.pdf")

    # Multiple files
    files = Attachments("code.py", "docs.md", "diagram.png")
    ```

    ### With prompt text

    ```python
    files = Attachments("analysis.csv").with_prompt("What trends do you see?")
    ```

    ### Integration with ChatBot

    ```python
    import talk_box as tb

    bot = tb.ChatBot().model("gpt-4-turbo")
    files = Attachments("codebase.py").with_prompt("Review this code")
    conversation = bot.chat(files)
    ```
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

        This method enables an interface for combining prompt text with file attachments, following
        the framework's chainable API design.

        Parameters
        ----------
        prompt
            The text prompt to include with the file attachments.

        Returns
        -------
        Attachments
            Returns self for method chaining.

        Examples
        --------
        ```python
        files = (Attachments("data.csv", "analysis.py")
                .with_prompt("Review this data analysis code and results"))
        ```
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
        Convert to chatlas-compatible content list for `Chat.chat()`.

        This method produces a list of content objects that can be passed directly to chatlas
        `Chat.chat()` method, combining the prompt text (if provided) with processed file contents.

        Returns
        -------
        List[Any]
            List of content objects: strings for text, ContentImageInline for images,
            ContentPDF for PDFs, etc.

        Examples
        --------
        ```python
        files = Attachments("image.png", "doc.pdf").with_prompt("Analyze these")
        contents = files.to_chat_contents()
        # Result: ["Analyze these", ContentImageInline(...), ContentPDF(...)]
        ```
        """
        contents = []

        # Add prompt text first if provided
        if self._prompt_text:
            contents.append(self._prompt_text)

        # Add processed file contents
        contents.extend(self._process_files())

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
        files = Attachments("code.py", "missing.txt", "diagram.png")
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
