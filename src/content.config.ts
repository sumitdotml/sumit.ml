import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const dateLabelsSchema = z
	.object({
		published: z.string().optional(),
		updated: z.string().optional(),
		page: z.string().optional(),
	})
	.optional();

const blog = defineCollection({
	loader: glob({ base: "./src/content/blog", pattern: "**/*.{md,mdx}" }),
	schema: ({ image }) =>
		z.object({
			title: z.string(),
			description: z.string(),
			breadcrumbTitle: z.string().optional(),
			pubDate: z.coerce.date(),
			updatedDate: z.coerce.date().optional(),
			dateLabels: dateLabelsSchema,
			image: image().optional(),
		}),
});

const pages = defineCollection({
	loader: glob({ base: "./src/content/pages", pattern: "**/*.{md,mdx}" }),
	schema: ({ image }) =>
		z.object({
			title: z.string(),
			description: z.string().optional(),
			breadcrumbTitle: z.string().optional(),
			date: z.coerce.date().optional(),
			dateLabel: z.string().optional(),
			dateLabels: dateLabelsSchema,
			image: image().optional(),
		}),
});

const research = defineCollection({
	loader: glob({ base: "./src/content/research", pattern: "**/*.{md,mdx}" }),
	schema: ({ image }) =>
		z.object({
			title: z.string(),
			subtitle: z.string().optional(),
			abstract: z.string().optional(),
			date: z.coerce.date(),
			authors: z
				.array(
					z.object({
						name: z.string(),
						affiliation: z.string().optional(),
						email: z.string().optional(),
					}),
				)
				.or(z.string())
				.optional(),
			keywords: z.array(z.string()).optional(),
			revisions: z
				.array(
					z.object({
						date: z.coerce.date(),
						note: z.string(),
					}),
				)
				.optional(),
			codeUrl: z.string().url().optional(),
			modelUrl: z.string().url().optional(),
			websiteUrl: z.string().url().optional(),
			logsUrl: z.string().url().optional(),
			pdfUrl: z.string().url().optional(),
			description: z.string().optional(),
			image: image().optional(),
			draft: z.boolean().optional().default(false),
		}),
});

export const collections = { blog, pages, research };
