export default function InputForm({ input, setInput, onSubmit }) {
  return (
    <div className="flex flex-col mb-6">
      <input
        type="text"
        className="border rounded-lg p-2 mb-3"
        placeholder="Nuevo mensaje"
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <button
        onClick={onSubmit}
        className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
      >
        Guardar Mensaje
      </button>
    </div>
  );
}
