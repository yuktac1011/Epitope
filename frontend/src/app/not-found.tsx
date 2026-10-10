import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#070b12] text-slate-200 p-6 text-center">
      <h1 className="text-4xl font-bold mb-3 text-emerald-400">404</h1>
      <h2 className="text-xl font-semibold mb-2">Page Not Found</h2>
      <p className="text-slate-400 max-w-md mb-6 text-sm">
        The requested console route does not exist.
      </p>
      <Link
        href="/"
        className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium transition-colors"
      >
        Return to FedShield Console
      </Link>
    </div>
  );
}
