// src/pages/Dashboard.jsx
import { useState, useEffect } from "react";
import { useBlockchain } from "../context/BlockchainContext";

export default function Dashboard() {
  const { 
    isConnected, 
    account, 
    message, 
    allMessages, 
    isLoading,
    refreshData 
  } = useBlockchain();
  
  const [stats, setStats] = useState({
    totalMessages: 0,
    lastUpdate: null
  });

  // Actualizar estadísticas cuando cambien los mensajes
  useEffect(() => {
    if (allMessages && allMessages.length > 0) {
      setStats({
        totalMessages: allMessages.length,
        lastUpdate: allMessages[0].timestamp
      });
    } else {
      setStats({
        totalMessages: 0,
        lastUpdate: null
      });
    }
  }, [allMessages]);



  if (!isConnected) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-center py-20">
            <h1 className="text-4xl font-bold text-white mb-4">Dashboard</h1>
            <p className="text-gray-300 text-lg">Conecta tu wallet para ver todas las transacciones y mensajes</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-2">Dashboard</h1>
          <p className="text-gray-300">Visualiza todos los mensajes almacenados en la blockchain</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6">
            <div className="flex flex-col">
              <h3 className="text-sm font-medium text-gray-300 uppercase tracking-wide">Total Mensajes</h3>
              <p className="text-3xl font-bold text-white mt-2">{stats.totalMessages}</p>
            </div>
          </div>
          
          <div className="glass rounded-2xl p-6">
            <div className="flex flex-col">
              <h3 className="text-sm font-medium text-gray-300 uppercase tracking-wide">Estado</h3>
              <p className="text-sm font-medium text-green-400 mt-2">Wallet Conectada</p>
              <p className="text-xs text-gray-400">{account?.slice(0, 6)}...{account?.slice(-4)}</p>
            </div>
          </div>
        </div>

        {/* Current Message */}
        <div className="glass rounded-2xl p-8">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-2xl font-bold text-white">Mensaje Actual</h2>
              <p className="text-gray-400 mt-2">El último mensaje almacenado en el contrato</p>
            </div>
            <button
              onClick={refreshData}
              disabled={isLoading}
              className="bg-blue-500 hover:bg-blue-600 disabled:opacity-50 text-white px-4 py-2 rounded-lg font-medium transition-colors duration-200 text-sm"
            >
              {isLoading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>Actualizando...</span>
                </div>
              ) : (
                'Actualizar'
              )}
            </button>
          </div>
          
          <div className="bg-slate-800/50 rounded-xl p-6">
            <p className="text-gray-300 text-lg">
              {message || "No hay mensaje guardado aún."}
            </p>
          </div>
        </div>

        {/* All Messages History */}
        <div className="glass rounded-2xl p-8">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-white">Historial Completo de Mensajes</h2>
            <p className="text-gray-400 mt-2">Todos los mensajes enviados al contrato desde su creación</p>
          </div>
          
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {isLoading && allMessages.length === 0 ? (
              <div className="text-center py-12">
                <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-gray-400">Cargando mensajes...</p>
              </div>
            ) : allMessages.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-gray-400 text-lg">No hay mensajes registrados aún.</p>
                <p className="text-gray-500 text-sm mt-2">Ve a la página de inicio para enviar tu primer mensaje</p>
              </div>
            ) : (
              allMessages.map((msg, index) => (
                <div key={`${msg.transactionHash}-${index}`} className="bg-slate-800/30 rounded-xl p-6 border border-slate-700">
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex-1">
                      <p className="text-white text-lg font-medium mb-3">{msg.message}</p>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm text-gray-400">
                        <span>👤 {msg.from.slice(0, 6)}...{msg.from.slice(-4)}</span>
                        <span>📅 {msg.timestamp ? msg.timestamp.toLocaleString() : msg.date}</span>
                        <span>🧱 Bloque: {msg.blockNumber}</span>
                      </div>
                    </div>
                    <div className="text-xs text-gray-500 bg-slate-700 px-2 py-1 rounded">
                      #{allMessages.length - index}
                    </div>
                  </div>
                  
                  <div className="text-xs text-blue-400 font-mono truncate bg-slate-900/50 p-2 rounded mt-3">
                    🔗 {msg.transactionHash}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
