import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import react from "@astrojs/react";
import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import type { Nodes, Root } from "mdast";
import type { Plugin, Transformer } from "unified";

const prettyArrow = "⇒";
const plainArrowPattern = /(^|[^-])->(?!>)/g;

function replaceTextArrows(node: Nodes): void {
	if (node.type === "text" && typeof node.value === "string") {
		node.value = node.value.replace(plainArrowPattern, `$1${prettyArrow}`);
		return;
	}

	if ("children" in node && Array.isArray(node.children)) {
		node.children.forEach(replaceTextArrows);
	}
}

const remarkPrettyArrows: Plugin<[], Root> = () => {
	const transformer: Transformer<Root, Root> = (tree, file) => {
		const filePath = String(file.path ?? file.history?.[0] ?? "");

		if (filePath.includes("/src/content/research/")) {
			return;
		}

		replaceTextArrows(tree);
	};

	return transformer;
};

// https://astro.build/config
export default defineConfig({
	site: "https://sumit.ml",
	integrations: [
		mdx({
			remarkPlugins: [remarkMath, remarkPrettyArrows],
			rehypePlugins: [rehypeKatex],
		}),
		sitemap({
			filter: (page) => !page.includes("/chiya"),
		}),
		react(),
	],
	markdown: {
		shikiConfig: {
			themes: {
				light: "catppuccin-latte",
				dark: "catppuccin-mocha",
			},
			defaultColor: false,
			wrap: false,
		},
		remarkPlugins: [remarkMath, remarkPrettyArrows],
		rehypePlugins: [rehypeKatex],
	},
});
