import { useEffect } from 'react'
import { useBlockchain } from '../context/BlockchainContext'
import TransactionCard from '../components/TransactionCard'
import Loader from '../components/Loader'

const Transaction = () => {
  const { account, transactions, loadTransactions, isLoading } = useBlockchain()

  useEffect(() => {
    if (account) {
      loadTransactions()
    }
  }, [account])

  if (!account) {
    return (
      <div className="max-w-4xl mx-auto text-center py-12">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <h2 className="text-2xl font-bold text-yellow-800 mb-4">
            🔒 Wallet no conectada
          </h2>
          <p className="text-yellow-700">
            Por favor conecta tu wallet para ver las transacciones
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          📋 Transacciones
        </h1>
        <p className="text-gray-600">
          Historial de todas las transacciones almacenadas en el contrato
        </p>
      </div>

      {/* Stats Section */}
      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-2xl font-bold text-blue-600 mb-2">
            {transactions.length}
          </div>
          <div className="text-sm text-gray-600">Total Transacciones</div>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-2xl font-bold text-green-600 mb-2">
            {transactions.length > 0 ? '✅' : '⏳'}
          </div>
          <div className="text-sm text-gray-600">
            {transactions.length > 0 ? 'Datos Cargados' : 'Cargando...'}
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-2xl font-bold text-purple-600 mb-2">
            🔗
          </div>
          <div className="text-sm text-gray-600">Blockchain Activa</div>
        </div>
      </div>

      {/* Refresh Button */}
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-gray-800">
          Historial de Transacciones
        </h2>
        <button
          onClick={loadTransactions}
          disabled={isLoading}
          className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2"
        >
          {isLoading ? <Loader size="sm" color="white" /> : null}
          <span>{isLoading ? 'Cargando...' : '🔄 Actualizar'}</span>
        </button>
      </div>

      {/* Transactions List */}
      {isLoading ? (
        <div className="flex justify-center items-center py-12">
          <div className="text-center">
            <Loader size="lg" />
            <p className="text-gray-600 mt-4">Cargando transacciones...</p>
          </div>
        </div>
      ) : transactions.length === 0 ? (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-12 text-center">
          <div className="text-6xl mb-4">📭</div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">
            No hay transacciones
          </h3>
          <p className="text-gray-600 mb-6">
            Las transacciones que crees en el Dashboard aparecerán aquí
          </p>
          <a
            href="/dashboard"
            className="inline-block bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium transition-colors"
          >
            Ir al Dashboard
          </a>
        </div>
      ) : (
        <div className="space-y-6">
          {transactions.map((transaction, index) => (
            <TransactionCard
              key={transaction.id}
              transaction={transaction}
              index={index}
            />
          ))}
        </div>
      )}

      {/* Footer Info */}
      {transactions.length > 0 && (
        <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center justify-between text-sm text-blue-700">
            <span>📊 Mostrando {transactions.length} transacciones</span>
            <span>🔗 Red: Localhost (Hardhat)</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default Transaction
