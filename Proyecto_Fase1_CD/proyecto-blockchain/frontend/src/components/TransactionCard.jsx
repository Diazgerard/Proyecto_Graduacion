import { useState } from "react";

export default function TransactionCard({ transaction, index }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const formatEther = (wei) => {
    try {
      return (Number(wei) / 1e18).toFixed(6);
    } catch {
      return '0.000000';
    }
  };

  const formatGwei = (wei) => {
    try {
      return (Number(wei) / 1e9).toFixed(2);
    } catch {
      return '0.00';
    }
  };

  const getStatusColor = (status) => {
    return status === 'success' ? 'text-green-400' : 'text-red-400';
  };

  const getStatusBadge = (status) => {
    const baseClasses = "px-2 py-1 rounded-full text-xs font-medium";
    return status === 'success' 
      ? `${baseClasses} bg-green-500/20 text-green-400`
      : `${baseClasses} bg-red-500/20 text-red-400`;
  };

  return (
    <div className="bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
      <div className="p-6">
        {/* Header */}
        <div className="flex justify-between items-start mb-4">
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-2">
              <span className={getStatusBadge(transaction.status)}>
                {transaction.status === 'success' ? '✓ Exitosa' : '✗ Fallida'}
              </span>
              <span className="text-xs text-gray-500 bg-slate-600 px-2 py-1 rounded">
                #{index + 1}
              </span>
            </div>
            <p className="text-white font-medium text-lg mb-2">{transaction.data}</p>
            <p className="text-xs text-gray-400">
              {transaction.timestamp.toLocaleString()}
            </p>
          </div>
        </div>

        {/* Basic Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <p className="text-xs text-gray-400 mb-1">Hash de Transacción</p>
            <p className="text-blue-400 font-mono text-sm truncate">{transaction.hash}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400 mb-1">Bloque</p>
            <p className="text-white text-sm">#{transaction.blockNumber}</p>
          </div>
        </div>

        {/* Toggle Button */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full text-blue-400 hover:text-blue-300 text-sm font-medium py-2 border-t border-slate-700 transition-colors"
        >
          {isExpanded ? 'Mostrar Menos ↑' : 'Ver Detalles Completos ↓'}
        </button>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="bg-slate-900/50 p-6 border-t border-slate-700">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Transaction Details */}
            <div className="space-y-4">
              <h4 className="text-white font-semibold mb-3">Detalles de Transacción</h4>
              
              <div>
                <p className="text-xs text-gray-400 mb-1">De</p>
                <p className="text-white font-mono text-sm break-all">{transaction.from}</p>
              </div>
              
              <div>
                <p className="text-xs text-gray-400 mb-1">Para (Contrato)</p>
                <p className="text-white font-mono text-sm break-all">{transaction.to}</p>
              </div>
              
              <div>
                <p className="text-xs text-gray-400 mb-1">Valor</p>
                <p className="text-white text-sm">{formatEther(transaction.value)} ETH</p>
              </div>
              
              <div>
                <p className="text-xs text-gray-400 mb-1">Nonce</p>
                <p className="text-white text-sm">{transaction.nonce}</p>
              </div>
            </div>

            {/* Gas Details */}
            <div className="space-y-4">
              <h4 className="text-white font-semibold mb-3">Información de Gas</h4>
              
              <div>
                <p className="text-xs text-gray-400 mb-1">Precio del Gas</p>
                <p className="text-white text-sm">{formatGwei(transaction.gasPrice)} Gwei</p>
              </div>
              
              <div>
                <p className="text-xs text-gray-400 mb-1">Límite de Gas</p>
                <p className="text-white text-sm">{Number(transaction.gasLimit).toLocaleString()}</p>
              </div>
              
              <div>
                <p className="text-xs text-gray-400 mb-1">Gas Usado</p>
                <p className="text-white text-sm">{Number(transaction.gasUsed).toLocaleString()}</p>
              </div>
              
              <div>
                <p className="text-xs text-gray-400 mb-1">Eficiencia de Gas</p>
                <p className="text-white text-sm">
                  {((Number(transaction.gasUsed) / Number(transaction.gasLimit)) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          {/* Block Details */}
          <div className="mt-6 pt-4 border-t border-slate-600">
            <h4 className="text-white font-semibold mb-3">Información del Bloque</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-xs text-gray-400 mb-1">Hash del Bloque</p>
                <p className="text-blue-400 font-mono text-sm truncate">{transaction.blockHash}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400 mb-1">Índice en Bloque</p>
                <p className="text-white text-sm">{transaction.transactionIndex}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400 mb-1">Confirmaciones</p>
                <p className="text-green-400 text-sm">{transaction.confirmations}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
