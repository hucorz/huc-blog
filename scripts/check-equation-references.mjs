import assert from "node:assert/strict";
import remarkEquationReferences from "../src/utils/remarkEquationReferences.mjs";

const source = String.raw`x=1
\label{eq:test}`;

const mathNode = {
  type: "math",
  value: source,
  data: {
    hName: "pre",
    hChildren: [
      {
        type: "element",
        tagName: "code",
        properties: { className: ["language-math", "math-display"] },
        children: [{ type: "text", value: source }],
      },
    ],
  },
};

const tree = {
  type: "root",
  children: [
    {
      type: "paragraph",
      children: [
        { type: "text", value: String.raw`由公式 \eqref{eq:test} 可知。` },
      ],
    },
    mathNode,
  ],
};

remarkEquationReferences()(tree, {});

const paragraph = tree.children[0];
const anchor = tree.children[1];
const numberedMath = tree.children[2];
const renderedMathSource = numberedMath.data.hChildren[0].children[0].value;

assert.equal(anchor.type, "html");
assert.match(anchor.value, /id="equation-eq-test"/);
assert.doesNotMatch(numberedMath.value, /\\label/);
assert.doesNotMatch(renderedMathSource, /\\label/);
assert.match(numberedMath.value, /\\tag\{1\}/);
assert.match(renderedMathSource, /\\tag\{1\}/);

const reference = paragraph.children.find(node => node.type === "link");
assert.equal(reference.url, "#equation-eq-test");
assert.equal(reference.children[0].value, "(1)");

const missingReferenceTree = {
  type: "root",
  children: [
    {
      type: "paragraph",
      children: [{ type: "text", value: String.raw`\eqref{eq:missing}` }],
    },
  ],
};

assert.throws(
  () => remarkEquationReferences()(missingReferenceTree, {}),
  /找不到公式引用/
);

