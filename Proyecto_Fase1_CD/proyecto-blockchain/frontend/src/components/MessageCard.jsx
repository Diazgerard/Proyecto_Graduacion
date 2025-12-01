export default function MessageCard({ message }) {
  return (
    <div className="bg-white p-4 rounded-xl shadow-md mb-4">
      <h2 className="text-lg font-bold mb-2">Mensaje almacenado</h2>
      <p className="text-gray-600">{message || "—"}</p>
    </div>
  );
}
