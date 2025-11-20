import { useState, useEffect } from 'react'
import { useBlockchain } from '../context/BlockchainContext'
import Loader from '../components/Loader'

const Dashboard = () => {
  const { 
    account, 
    contract, 
    isLoading, 
    getData, 
    setData, 
    storeTransaction, 
    executeAction 
  } = useBlockchain()

  const [currentData, setCurrentData] = useState('')
  const [newData, setNewData] = useState('')
  const [transactionForm, setTransactionForm] = useState({
    to: '',
    amount: '',
    message: ''
  })
  const [actionInput, setActionInput] = useState('')

  // Cargar datos actuales al montar el componente
  useEffect(() => {
    if (contract) {
      loadCurrentData()
    }
  }, [contract])

  const loadCurrentData = async () => {
    const data = await getData()
    if (data) {
      setCurrentData(data)
    }
  }

  const handleSetData = async (e) => {
    e.preventDefault()
    if (!newData.trim()) return

    const success = await setData(newData)
    if (success) {
      setNewData('')
      loadCurrentData()
    }
  }

  const handleStoreTransaction = async (e) => {
    e.preventDefault()
    const { to, amount, message } = transactionForm
    
    if (!to || !amount || !message) {
      alert('Por favor completa todos los campos')
      return
    }

    const success = await storeTransaction(to, amount, message)
    if (success) {
      setTransactionForm({ to: '', amount: '', message: '' })
    }
  }

  const handleExecuteAction = async (e) => {
    e.preventDefault()
    if (!actionInput.trim()) return

    const result = await executeAction(actionInput)
    if (result) {
      setActionInput('')
      loadCurrentData()
    }
  }

  if (!account) {
    return (
      <div className="min-h-screen flex items-center justify-center px-6">
        <div className="glass rounded-2xl p-12 text-center max-w-md mx-auto">
          <div className="w-20 h-20 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-2xl mx-auto mb-6"></div>
          <h2 className="text-2xl font-bold text-white mb-4">
            Wallet Requerida
          </h2>
          <p className="text-gray-300 text-lg mb-6">
            Conecta tu wallet MetaMask para acceder al dashboard y todas las funcionalidades de la dApp.
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
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Control Center
          </h1>
          <p className="text-gray-300 text-lg">
            Gestiona tu interacción con la blockchain
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <div className="glass rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm mb-2">Estado del Contrato</p>
                <p className="text-xl font-bold text-white">
                  {contract ? 'Conectado' : 'Desconectado'}
                </p>
              </div>
              <div className={`w-12 h-12 rounded-xl ${contract ? 'bg-green-500' : 'bg-red-500'}`}></div>
            </div>
          </div>

          <div className="glass rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm mb-2">Red Activa</p>
                <p className="text-xl font-bold text-white">Localhost</p>
              </div>
              <div className="w-12 h-12 bg-purple-500 rounded-xl"></div>
            </div>
          </div>

          <div className="glass rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm mb-2">Tu Wallet</p>
                <p className="text-lg font-bold text-white font-mono">
                  {account?.slice(0, 6)}...{account?.slice(-4)}
                </p>
              </div>
              <div className="w-12 h-12 bg-blue-500 rounded-xl"></div>
            </div>
          </div>
        </div>

        {/* Main Sections */}
        <div className="grid lg:grid-cols-2 gap-4 mb-4">
          {/* Contract Data Section */}
          <div className="glass rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-3">
              <div className="w-6 h-6 bg-gradient-to-r from-blue-400 to-purple-500 rounded-md"></div>
              <h2 className="text-sm font-bold text-white">Datos del Contrato</h2>
            </div>
            
            <div className="mb-6">
              <label className="block text-gray-300 text-sm font-medium mb-3">
                Datos actuales almacenados:
              </label>
              <div className="bg-white/5 border border-white/10 rounded-lg p-4">
                <code className="text-white font-mono text-sm break-all">
                  {currentData || 'Cargando datos...'}
                </code>
              </div>
            </div>

            <form onSubmit={handleSetData} className="space-y-4">
              <div>
                <label htmlFor="newData" className="block text-gray-300 text-sm font-medium mb-2">
                  Actualizar datos:
                </label>
                <input
                  type="text"
                  id="newData"
                  value={newData}
                  onChange={(e) => setNewData(e.target.value)}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all text-base"
                  placeholder="Ingresa nuevos datos..."
                  disabled={isLoading}
                />
              </div>
              <button
                type="submit"
                disabled={isLoading || !newData.trim()}
                className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white py-4 px-6 rounded-lg font-medium transition-all duration-300 transform hover:scale-[1.02] flex items-center justify-center space-x-2 text-base"
              >
                {isLoading && (
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2"></div>
                )}
                <span>{isLoading ? 'Actualizando...' : 'Actualizar Datos'}</span>
              </button>
            </form>
          </div>

          {/* Store Transaction Section */}
          <div className="glass rounded-2xl p-8">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-white">Transacción</h2>
            </div>
            
            <form onSubmit={handleStoreTransaction} className="space-y-6">
              <div>
                <label htmlFor="to" className="block text-gray-300 text-sm font-medium mb-2">
                  Dirección destino:
                </label>
                <input
                  type="text"
                  id="to"
                  value={transactionForm.to}
                  onChange={(e) => setTransactionForm(prev => ({ ...prev, to: e.target.value }))}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all font-mono text-base"
                  placeholder="0x..."
                  disabled={isLoading}
                />
              </div>
              
              <div>
                <label htmlFor="amount" className="block text-gray-300 text-sm font-medium mb-2">
                  Cantidad (ETH):
                </label>
                <input
                  type="number"
                  id="amount"
                  value={transactionForm.amount}
                  onChange={(e) => setTransactionForm(prev => ({ ...prev, amount: e.target.value }))}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all text-base"
                  placeholder="0.0"
                  step="0.01"
                  disabled={isLoading}
                />
              </div>
              
              <div>
                <label htmlFor="message" className="block text-gray-300 text-sm font-medium mb-2">
                  Mensaje:
                </label>
                <textarea
                  id="message"
                  value={transactionForm.message}
                  onChange={(e) => setTransactionForm(prev => ({ ...prev, message: e.target.value }))}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all resize-none text-base"
                  placeholder="Descripción de la transacción..."
                  rows="3"
                  disabled={isLoading}
                />
              </div>
              
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white py-4 px-6 rounded-lg font-medium transition-all duration-300 transform hover:scale-[1.02] flex items-center justify-center space-x-2 text-base"
              >
                {isLoading && (
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2"></div>
                )}
                <span>{isLoading ? 'Procesando...' : 'Crear Transacción'}</span>
              </button>
            </form>
          </div>
        </div>

        {/* Execute Action Section */}
        <div className="glass rounded-2xl p-8">
          <div className="mb-6">
            <h2 className="text-xl font-bold text-white">Ejecutar Acción Personalizada</h2>
          </div>
          
          <form onSubmit={handleExecuteAction} className="space-y-6">
            <div>
              <label htmlFor="action" className="block text-gray-300 text-sm font-medium mb-2">
                Describe la acción a ejecutar:
              </label>
              <input
                type="text"
                id="action"
                value={actionInput}
                onChange={(e) => setActionInput(e.target.value)}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all text-base"
                placeholder="Ej: actualizar configuración, procesar datos, etc..."
                disabled={isLoading}
              />
            </div>
            
            <button
              type="submit"
              disabled={isLoading || !actionInput.trim()}
              className="w-full bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed text-white py-4 px-6 rounded-lg font-medium transition-all duration-300 transform hover:scale-[1.02] flex items-center justify-center space-x-2 text-base"
            >
              {isLoading && (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2"></div>
              )}
              <span>{isLoading ? 'Ejecutando...' : 'Ejecutar Acción'}</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
