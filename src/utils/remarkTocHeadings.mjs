/**
 * Preserve clean heading text before rehype-katex expands inline math into
 * MathML and HTML. Astro's generated heading text otherwise contains all of
 * KaTeX's accessibility and fallback layers at once.
 */
export default function remarkTocHeadings() {
  return (tree, file) => {
    const tocHeadingSources = [];

    visitHeadings(tree, node => {
      tocHeadingSources.push({
        depth: node.depth,
        text: serializeInlineContent(node.children),
      });
    });

    file.data.astro ??= {};
    file.data.astro.frontmatter ??= {};
    file.data.astro.frontmatter.tocHeadingSources = tocHeadingSources;
  };
}

function visitHeadings(node, visitor) {
  if (node.type === "heading") visitor(node);
  if (!Array.isArray(node.children)) return;

  for (const child of node.children) {
    visitHeadings(child, visitor);
  }
}

function serializeInlineContent(nodes) {
  return nodes.map(serializeNode).join("");
}

function serializeNode(node) {
  if (node.type === "inlineMath") return `$${node.value}$`;
  if (node.type === "text" || node.type === "inlineCode") return node.value;
  if (node.type === "image") return node.alt ?? "";
  if (node.type === "break") return " ";
  if (Array.isArray(node.children))
    return serializeInlineContent(node.children);
  return "";
}
