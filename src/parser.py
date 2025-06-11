from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser.file.markdown import MarkdownNodeParser
from llama_index.core import Settings


def markdownParser(input_dir, embed_model= None, required_exts=[".md"], header_path_separator="/"):
    """
    Loads documents from a directory, parses them into nodes,
    converts node text to lowercase, and generates embeddings for each node.

    Args:
        input_dir (str): The directory to read documents from.
        embed_model: The embedding model to use for generating node embeddings.
        required_exts (list, optional): List of required file extensions. Defaults to [".md"].
        header_path_separator (str, optional): Separator for markdown headers. Defaults to "/".

    Returns:
        list: A list of processed nodes with embeddings.
    """
    documents = SimpleDirectoryReader(input_dir=input_dir, required_exts=required_exts).load_data()
    parser = MarkdownNodeParser(header_path_separator=header_path_separator)
    nodes = parser.get_nodes_from_documents(documents)

    if not embed_model:
        embed_model = Settings.embed_model  # Use the default embed model if not provided
    processed_nodes = []
    for node in nodes:
        # Lowercase the text
        node.text = node.text.lower()
        # Remove file name from the node's metadata
        if "file_path" in node.metadata:
            # get only filename from file_path
            filepath = nodes[1].metadata.get("file_path", "")
            nodes[1].metadata["file_path"] = filepath.split("\\")[-1] if filepath else ""

        # Generate and assign embedding
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")  # Use get_content for embedding
        )
        node.embedding = node_embedding
        processed_nodes.append(node)

    return processed_nodes

### NOTE:
# modify the function in MarkdownNodeParser to get the node headers
# def _build_node_from_split(
#         self,
#         text_split: str,
#         node: BaseNode,
#         header_stack: List[tuple[int, str]],
#     ) -> TextNode:
#         """Build node from single text split."""
#         node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]

#         if self.include_metadata:
#             separator = self.header_path_separator
#             if header_stack:
#                 # Build the full path including all headers in the stack
#                 header_path = separator + separator.join(h[1] for h in header_stack)
#             else:
#                 # Root level content (no headers)
#                 header_path = separator
            
#             node.metadata["header_path"] = header_path

#         return node"""