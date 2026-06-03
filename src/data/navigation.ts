export type NavLink = {
	href: string;
	label: string;
};

export const NAV_LINKS: NavLink[] = [
	{ href: "/about", label: "About" },
	{ href: "/blog", label: "Blog" },
	{ href: "/research", label: "Research" },
	{ href: "/weekly", label: "Weekly" },
];

export function findNavLabel(pathSegment: string): string | undefined {
	const normalized = `/${pathSegment}`;
	return NAV_LINKS.find((link) => link.href === normalized)?.label;
}
