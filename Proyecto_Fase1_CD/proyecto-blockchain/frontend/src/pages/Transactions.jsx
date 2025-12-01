import { useState, useEffect } from "react";
import { useBlockchain } from "../context/BlockchainContext";
import TransactionCard from "../components/TransactionCard";

export default function Transactions() {
  const { 
    contract, 
    isConnected, 
    transactions, 
    allMessages, 
    provider,
    isLoading, 
    refreshData 
  } = useBlockchain();
  
  const [enhancedTransactions, setEnhancedTransactions] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    successful: 0,
    failed: 0,
    totalGasUsed: 0
  });

  useEffect(() => {
    if (isConnected && transactions.length > 0 && contract && provider) {
      enhanceTransactionsWithDetails();
    }
  }, [transactions, isConnected, contract, provider]);

  const enhanceTransactionsWithDetails = async () => {
    if (!contract || !provider || transactions.length === 0) return;
    
    try {
      console.log(`🔍 Mejorando ${transactions.length} transacciones con detalles completos...`);
      
      // Obtener detalles completos de cada transacción
      const enhancedDetails = await Promise.all(
        transactions.map(async (tx) => {
          try {
            const [txDetails, receipt, currentBlockNumber] = await Promise.all([
              provider.getTransaction(tx.transactionHash),
              provider.getTransactionReceipt(tx.transactionHash),
              provider.getBlockNumber()
            ]);
            
            const contractAddress = await contract.getAddress();
            
            return {
              // Datos básicos del contexto
              hash: tx.transactionHash,
              from: tx.from,
              to: contractAddress,
              data: tx.message,
              timestamp: tx.timestamp,
              blockNumber: tx.blockNumber,
              
              // Detalles de la transacción
              value: txDetails.value,
              gasPrice: txDetails.gasPrice,
              gasLimit: txDetails.gasLimit,
              gasUsed: receipt.gasUsed,
              status: receipt.status === 1 ? 'success' : 'failed',
              blockHash: receipt.blockHash,
              nonce: txDetails.nonce,
              transactionIndex: receipt.transactionIndex,
              confirmations: currentBlockNumber - tx.blockNumber + 1,
              type: txDetails.type || 0,
              
              // Cálculos adicionales
              gasEfficiency: (Number(receipt.gasUsed) / Number(txDetails.gasLimit)) * 100,
              gasCost: Number(txDetails.gasPrice) * Number(receipt.gasUsed)
            };
          } catch (error) {
            console.error(`❌ Error mejorando transacción ${tx.transactionHash}:`, error);
            // Retornar datos básicos si falla la mejora
            return {
              hash: tx.transactionHash,
              from: tx.from,
              to: 'N/A',
              data: tx.message,
              timestamp: tx.timestamp,
              blockNumber: tx.blockNumber,
              status: 'success', // Asumimos éxito si está en los eventos
              gasUsed: 0,
              gasPrice: 0,
              value: 0
            };
          }
        })
      );
      
      setEnhancedTransactions(enhancedDetails);
      
      // Calcular estadísticas
      const successful = enhancedDetails.filter(tx => tx.status === 'success').length;
      const failed = enhancedDetails.filter(tx => tx.status === 'failed').length;
      const totalGasUsed = enhancedDetails.reduce((sum, tx) => sum + Number(tx.gasUsed || 0), 0);
      
      setStats({
        total: enhancedDetails.length,
        successful,
        failed,
        totalGasUsed
      });
      
      console.log(`✅ ${enhancedDetails.length} transacciones mejoradas correctamente`);
      
    } catch (error) {
      console.error("❌ Error mejorando transacciones:", error);
    }
  };

  if (!isConnected) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-center py-20">
            <h1 className="text-4xl font-bold text-white mb-4">Transacciones</h1>
            <p className="text-gray-300 text-lg">Conecta tu wallet para ver todas las transacciones</p>
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
          <h1 className="text-4xl font-bold text-white mb-2">Transacciones</h1>
          <p className="text-gray-300">Historial completo de transacciones del contrato</p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="glass rounded-2xl p-6">
            <h3 className="text-sm font-medium text-gray-300 uppercase tracking-wide">Total</h3>
            <p className="text-3xl font-bold text-white mt-2">{stats.total}</p>
          </div>
          
          <div className="glass rounded-2xl p-6">
            <h3 className="text-sm font-medium text-gray-300 uppercase tracking-wide">Exitosas</h3>
            <p className="text-3xl font-bold text-green-400 mt-2">{stats.successful}</p>
          </div>
          
          <div className="glass rounded-2xl p-6">
            <h3 className="text-sm font-medium text-gray-300 uppercase tracking-wide">Fallidas</h3>
            <p className="text-3xl font-bold text-red-400 mt-2">{stats.failed}</p>
          </div>
          
          <div className="glass rounded-2xl p-6">
            <h3 className="text-sm font-medium text-gray-300 uppercase tracking-wide">Gas Total</h3>
            <p className="text-2xl font-bold text-blue-400 mt-2">{stats.totalGasUsed.toLocaleString()}</p>
          </div>
        </div>

        {/* Transactions List */}
        <div className="glass rounded-2xl p-8">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-white">Historial de Transacciones</h2>
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

          {isLoading && enhancedTransactions.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-gray-400">Cargando transacciones...</p>
            </div>
          ) : enhancedTransactions.length === 0 && transactions.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-400 text-lg">No hay transacciones registradas aún.</p>
              <p className="text-gray-500 text-sm mt-2">Las transacciones aparecerán aquí cuando interactúes con el contrato</p>
            </div>
          ) : enhancedTransactions.length === 0 && transactions.length > 0 ? (
            <div className="text-center py-12">
              <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-gray-400">Procesando detalles de transacciones...</p>
            </div>
          ) : (
            <div className="space-y-4">
              {enhancedTransactions.map((transaction, index) => (
                <TransactionCard key={transaction.hash} transaction={transaction} index={index} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
