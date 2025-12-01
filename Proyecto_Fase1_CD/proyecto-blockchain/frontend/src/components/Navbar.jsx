// src/components/Navbar.jsx
import { Link } from "react-router-dom";
import { useBlockchain } from "../context/BlockchainContext";

export default function Navbar() {
  const { account, isConnected, connectWallet, disconnectWallet } = useBlockchain();

  return (
    <nav className="w-full flex justify-between items-center bg-white shadow-md p-4 mb-6 rounded-xl">
      <h1 className="text-2xl font-bold text-gray-800">Proyecto Blockchain</h1>

      <div className="flex items-center gap-3">
        <Link to="/" className="text-gray-700 hover:text-gray-900">
          Home
        </Link>
        <Link to="/dashboard" className="text-gray-700 hover:text-gray-900">
          Dashboard
        </Link>
        <Link to="/transactions" className="text-gray-700 hover:text-gray-900">
          Transacciones
        </Link>

        {!isConnected ? (
          <button
            onClick={connectWallet} // Llama a la función de tu contexto
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Conectar Wallet
          </button>
        ) : (
          <>
            <span className="text-gray-700 font-semibold">
              {account.slice(0, 6)}...{account.slice(-4)}
            </span>
            <button
              onClick={disconnectWallet} // Llama al desconectar de tu contexto
              className="px-3 py-1 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Desconectar
            </button>
          </>
        )}
      </div>
    </nav>
  );
}
