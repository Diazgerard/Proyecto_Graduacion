import { useState, useEffect } from 'react'
import { useBlockchain } from '../context/BlockchainContext'
import TransactionCard from '../components/TransactionCard'

const Transactions = () => {
  const { account, getAllTransactions, isLoading } = useBlockchain()
  const [transactions, setTransactions] = useState([])
  const [filter, setFilter] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    if (account) {
      loadTransactions()
    }
  }, [account])

  const loadTransactions = async () => {
    const txs = await getAllTransactions()
    if (txs) {
      setTransactions(txs)
    }
  }

  const filteredTransactions = transactions.filter(tx => {
    const matchesSearch = 
      tx.to.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tx.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tx.from.toLowerCase().includes(searchTerm.toLowerCase())
    
    if (filter === 'sent') {
      return tx.from.toLowerCase() === account?.toLowerCase() && matchesSearch
    } else if (filter === 'received') {
      return tx.to.toLowerCase() === account?.toLowerCase() && matchesSearch
    }
    
    return matchesSearch
  })

  if (!account) {
    return (
      <div className="min-h-screen flex items-center justify-center px-6">
        <div className="glass rounded-3xl p-12 text-center max-w-md mx-auto">
          <h2 className="text-2xl font-bold text-white mb-4">
            Wallet Requerida
          </h2>
          <p className="text-gray-300 mb-6">
            Conecta tu wallet MetaMask para ver el historial de transacciones.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen px-6 py-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">
            Historial de Transacciones
          </h1>
          <p className="text-gray-300 text-lg">
            Explora todas las transacciones registradas en la blockchain
          </p>
        </div>

        {/* Stats */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          <div className="glass rounded-2xl p-6 text-center">
            <p className="text-2xl font-bold text-white mb-1">{transactions.length}</p>
            <p className="text-gray-300 text-sm">Total</p>
          </div>

          <div className="glass rounded-2xl p-6 text-center">
            <p className="text-2xl font-bold text-white mb-1">
              {transactions.filter(tx => tx.from.toLowerCase() === account?.toLowerCase()).length}
            </p>
            <p className="text-gray-300 text-sm">Enviadas</p>
          </div>

          <div className="glass rounded-2xl p-6 text-center">
            <p className="text-2xl font-bold text-white mb-1">
              {transactions.filter(tx => tx.to.toLowerCase() === account?.toLowerCase()).length}
            </p>
            <p className="text-gray-300 text-sm">Recibidas</p>
          </div>

          <div className="glass rounded-2xl p-6 text-center">
            <p className="text-2xl font-bold text-white mb-1">
              {transactions.reduce((sum, tx) => sum + parseFloat(tx.amount || '0'), 0).toFixed(2)}
            </p>
            <p className="text-gray-300 text-sm">Total ETH</p>
          </div>
        </div>

        {/* Filters and Search */}
        <div className="glass rounded-2xl p-6 mb-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <input
                type="text"
                placeholder="Buscar por dirección o mensaje..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
              />
            </div>

            {/* Filter Buttons */}
            <div className="flex space-x-2">
              {[
                { key: 'all', label: 'Todas', icon: '📋' },
                { key: 'sent', label: 'Enviadas', icon: '📤' },
                { key: 'received', label: 'Recibidas', icon: '📥' }
              ].map((filterOption) => (
                <button
                  key={filterOption.key}
                  onClick={() => setFilter(filterOption.key)}
                  className={`px-4 py-2 rounded-xl font-medium transition-all duration-300 flex items-center space-x-2 ${
                    filter === filterOption.key
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                      : 'bg-white/5 text-gray-300 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <span>{filterOption.icon}</span>
                  <span>{filterOption.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Transactions List */}
        <div className="space-y-4">
          {isLoading ? (
            <div className="flex justify-center py-12">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 border-4 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span className="text-white text-lg">Cargando transacciones...</span>
              </div>
            </div>
          ) : filteredTransactions.length > 0 ? (
            <div className="grid gap-6">
              {filteredTransactions.map((transaction, index) => (
                <TransactionCard 
                  key={index} 
                  transaction={transaction} 
                  currentAccount={account}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="glass rounded-3xl p-12 max-w-md mx-auto">
                <h3 className="text-xl font-bold text-white mb-2">
                  {searchTerm ? 'Sin resultados' : 'No hay transacciones'}
                </h3>
                <p className="text-gray-300">
                  {searchTerm 
                    ? 'No se encontraron transacciones que coincidan con tu búsqueda.'
                    : 'Aún no hay transacciones registradas. ¡Crea tu primera transacción desde el Dashboard!'
                  }
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Refresh Button */}
        <div className="text-center mt-12">
          <button
            onClick={loadTransactions}
            disabled={isLoading}
            className="inline-flex items-center space-x-2 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 text-white px-6 py-3 rounded-xl font-semibold transition-all duration-300 transform hover:scale-105"
          >
            <span>{isLoading ? 'Actualizando...' : 'Actualizar Lista'}</span>
          </button>
        </div>
      </div>
    </div>
  )
}

export default Transactions