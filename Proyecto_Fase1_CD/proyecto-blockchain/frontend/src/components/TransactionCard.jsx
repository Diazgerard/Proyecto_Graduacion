const TransactionCard = ({ transaction, index, currentAccount }) => {
  const formatAddress = (address) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`
  }

  const formatAmount = (amount) => {
    return amount === '0' ? '0.00' : parseFloat(amount).toFixed(4)
  }

  const formatDate = (timestamp) => {
    if (!timestamp) return 'Fecha no disponible'
    try {
      return new Date(timestamp * 1000).toLocaleString('es-ES', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    } catch {
      return 'Fecha no disponible'
    }
  }

  const isOutgoing = transaction.from?.toLowerCase() === currentAccount?.toLowerCase()
  const isIncoming = transaction.to?.toLowerCase() === currentAccount?.toLowerCase()

  const getTransactionType = () => {
    if (isOutgoing) return { type: 'sent', label: 'Enviada', icon: '📤', color: 'from-red-400 to-pink-500' }
    if (isIncoming) return { type: 'received', label: 'Recibida', icon: '📥', color: 'from-green-400 to-blue-500' }
    return { type: 'other', label: 'Externa', icon: '🔄', color: 'from-gray-400 to-gray-600' }
  }

  const txType = getTransactionType()

  return (
    <div className="glass rounded-2xl p-6 hover:scale-[1.02] transition-all duration-300 group">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <div className={`w-12 h-12 bg-gradient-to-r ${txType.color} rounded-xl flex items-center justify-center text-white font-bold shadow-lg`}>
            <span className="text-lg">{txType.icon}</span>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">
              {txType.label}
            </h3>
            <p className="text-gray-300 text-sm">
              TX #{transaction.id || index}
            </p>
          </div>
        </div>
        
        <div className="text-right">
          <div className={`inline-flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium ${
            txType.type === 'sent' ? 'bg-red-500/20 text-red-300' :
            txType.type === 'received' ? 'bg-green-500/20 text-green-300' :
            'bg-gray-500/20 text-gray-300'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              txType.type === 'sent' ? 'bg-red-400' :
              txType.type === 'received' ? 'bg-green-400' :
              'bg-gray-400'
            } animate-pulse`}></div>
            <span>Confirmada</span>
          </div>
        </div>
      </div>

      {/* Transaction Details */}
      <div className="space-y-4">
        {/* Amount */}
        <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl">
          <span className="text-gray-300 font-medium">Cantidad</span>
          <div className="text-right">
            <span className={`text-xl font-bold ${
              txType.type === 'sent' ? 'text-red-300' :
              txType.type === 'received' ? 'text-green-300' :
              'text-white'
            }`}>
              {txType.type === 'sent' ? '-' : '+'}{formatAmount(transaction.amount)} ETH
            </span>
            <p className="text-gray-400 text-sm">
              ≈ ${(parseFloat(transaction.amount || '0') * 2000).toFixed(2)} USD
            </p>
          </div>
        </div>

        {/* Addresses */}
        <div className="grid md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <span className="text-gray-300 text-sm font-medium">Desde</span>
            <div className={`p-3 rounded-xl border ${
              isOutgoing ? 'bg-red-500/10 border-red-500/20' : 'bg-white/5 border-white/10'
            }`}>
              <div className="flex items-center space-x-2">
                {isOutgoing && (
                  <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                )}
                <span className="text-white font-mono text-sm">
                  {formatAddress(transaction.from)}
                </span>
                {isOutgoing && (
                  <span className="text-red-300 text-xs font-medium">(Tú)</span>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <span className="text-gray-300 text-sm font-medium">Hacia</span>
            <div className={`p-3 rounded-xl border ${
              isIncoming ? 'bg-green-500/10 border-green-500/20' : 'bg-white/5 border-white/10'
            }`}>
              <div className="flex items-center space-x-2">
                {isIncoming && (
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                )}
                <span className="text-white font-mono text-sm">
                  {formatAddress(transaction.to)}
                </span>
                {isIncoming && (
                  <span className="text-green-300 text-xs font-medium">(Tú)</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Message */}
        {transaction.message && (
          <div className="space-y-2">
            <span className="text-gray-300 text-sm font-medium">Mensaje</span>
            <div className="p-4 bg-white/5 rounded-xl border border-white/10">
              <p className="text-white italic">
                "{transaction.message}"
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="mt-6 pt-4 border-t border-white/10 flex items-center justify-between text-sm">
        <div className="text-gray-400">
          <span>{formatDate(transaction.timestamp)}</span>
        </div>
        
        <div className="text-gray-400">
          <span>Bloque #{(transaction.id || 0) + 1}</span>
        </div>
      </div>
    </div>
  )
}

export default TransactionCard
