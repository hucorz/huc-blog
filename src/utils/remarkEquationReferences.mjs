const LABEL_PATTERN = /\\label\s*\{([^{}]+)\}/g;
const REFERENCE_PATTERN = /\\(eqref|ref)\s*\{([^{}]+)\}/g;
const MANUAL_TAG_PATTERN = /\\tag\*?\s*\{/;
const VALID_LABEL_PATTERN = /^[A-Za-z][A-Za-z0-9:_-]*$/;

const REFERENCE_EXCLUDED_NODES = new Set([
  "code",
  "definition",
  "html",
  "inlineCode",
  "inlineMath",
  "link",
  "linkReference",
  "math",
]);

/**
 * Add automatic numbering and cross references to labelled display equations.
 *
 * Authoring syntax:
 *
 * $$
 * E = mc^2
 * \label{eq:mass-energy}
 * $$
 *
 * Later prose can use \eqref{eq:mass-energy} or \ref{eq:mass-energy}.
 * Only labelled display equations are numbered.
 */
export default function remarkEquationReferences() {
  return (tree, file) => {
    const labels = new Map();
    const ids = new Set();
    let equationNumber = 0;

    collectEquations(tree);
    replaceReferences(tree);

    function collectEquations(parent) {
      if (!Array.isArray(parent.children)) return;

      for (let index = 0; index < parent.children.length; index += 1) {
        const node = parent.children[index];

        if (node.type === "math" && typeof node.value === "string") {
          const labelMatches = [...node.value.matchAll(LABEL_PATTERN)];

          if (labelMatches.length > 1) {
            fail(file, "一个公式只能包含一个 \\label。", node);
          }

          if (labelMatches.length === 1) {
            const label = labelMatches[0][1].trim();

            if (!VALID_LABEL_PATTERN.test(label)) {
              fail(
                file,
                `公式标签“${label}”格式无效；请使用字母开头，并只包含字母、数字、冒号、下划线或连字符。`,
                node
              );
            }

            if (labels.has(label)) {
              fail(file, `公式标签“${label}”重复。`, node);
            }

            if (MANUAL_TAG_PATTERN.test(node.value)) {
              fail(
                file,
                `公式“${label}”同时使用了 \\label 和 \\tag；自动编号公式不需要手写 \\tag。`,
                node
              );
            }

            equationNumber += 1;
            const id = createEquationId(label);

            if (ids.has(id)) {
              fail(file, `公式标签“${label}”生成了重复的锚点“${id}”。`, node);
            }

            labels.set(label, { id, number: equationNumber });
            ids.add(id);

            const numberedValue = `${node.value.replace(LABEL_PATTERN, "").trimEnd()}\n\\tag{${equationNumber}}`;
            setMathValue(node, numberedValue);

            parent.children.splice(index, 0, {
              type: "html",
              value: `<span id="${id}" class="equation-anchor" aria-hidden="true"></span>`,
            });
            index += 1;
          }
        }

        collectEquations(node);
      }
    }

    function replaceReferences(parent) {
      if (!Array.isArray(parent.children)) return;

      const rewrittenChildren = [];

      for (const node of parent.children) {
        if (node.type === "text" && typeof node.value === "string") {
          rewrittenChildren.push(...rewriteTextNode(node, labels, file));
          continue;
        }

        if (!REFERENCE_EXCLUDED_NODES.has(node.type)) {
          replaceReferences(node);
        }

        rewrittenChildren.push(node);
      }

      parent.children = rewrittenChildren;
    }
  };
}

function rewriteTextNode(node, labels, file) {
  const matches = [...node.value.matchAll(REFERENCE_PATTERN)];
  if (matches.length === 0) return [node];

  const nodes = [];
  let cursor = 0;

  for (const match of matches) {
    const start = match.index;
    const end = start + match[0].length;
    const command = match[1];
    const label = match[2].trim();
    const target = labels.get(label);

    if (!target) {
      fail(file, `找不到公式引用“${label}”。`, node);
    }

    if (start > cursor) {
      nodes.push({ type: "text", value: node.value.slice(cursor, start) });
    }

    const visibleNumber =
      command === "eqref" ? `(${target.number})` : String(target.number);

    nodes.push({
      type: "link",
      url: `#${target.id}`,
      title: `跳转到公式（${target.number}）`,
      children: [{ type: "text", value: visibleNumber }],
      data: {
        hProperties: {
          className: ["equation-reference"],
          "aria-label": `跳转到公式（${target.number}）`,
        },
      },
    });

    cursor = end;
  }

  if (cursor < node.value.length) {
    nodes.push({ type: "text", value: node.value.slice(cursor) });
  }

  return nodes;
}

function createEquationId(label) {
  return `equation-${label.toLowerCase().replaceAll(":", "-")}`;
}

function setMathValue(node, value) {
  node.value = value;

  const displayCode = node.data?.hChildren?.find(
    child =>
      child.type === "element" &&
      Array.isArray(child.properties?.className) &&
      child.properties.className.includes("math-display")
  );

  if (displayCode) {
    displayCode.children = [{ type: "text", value }];
  }
}

function fail(file, message, node) {
  if (typeof file?.fail === "function") {
    file.fail(message, node);
  }

  const path = file?.path ? `${file.path}: ` : "";
  throw new Error(`${path}${message}`);
}
