import { createContext, useContext, useState, useEffect } from 'react'
import { ethers } from 'ethers'
import { CONTRACT_ADDRESS } from '../contracts/contract-address'
import contractABI from '../contracts/contractABI.json'

const BlockchainContext = createContext()

export const useBlockchain = () => {
  const context = useContext(BlockchainContext)
  if (!context) {
    throw new Error('useBlockchain must be used within a BlockchainProvider')
  }
  return context
}

export const BlockchainProvider = ({ children }) => {
  const [account, setAccount] = useState('')
  const [contract, setContract] = useState(null)
  const [provider, setProvider] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [transactions, setTransactions] = useState([])

  // Conectar wallet
  const connectWallet = async () => {
    try {
      if (window.ethereum) {
        setIsLoading(true)
        
        // Solicitar conexión a MetaMask
        const accounts = await window.ethereum.request({
          method: 'eth_requestAccounts'
        })
        
        // Configurar provider
        const web3Provider = new ethers.BrowserProvider(window.ethereum)
        const signer = await web3Provider.getSigner()
        
        // Instanciar contrato
        const contractInstance = new ethers.Contract(
          CONTRACT_ADDRESS,
          contractABI,
          signer
        )
        
        setAccount(accounts[0])
        setProvider(web3Provider)
        setContract(contractInstance)
        
        console.log('✅ Wallet conectada:', accounts[0])
        return true
      } else {
        alert('MetaMask no está instalado!')
        return false
      }
    } catch (error) {
      console.error('Error conectando wallet:', error)
      alert('Error al conectar wallet: ' + error.message)
      return false
    } finally {
      setIsLoading(false)
    }
  }

  // Desconectar wallet
  const disconnectWallet = () => {
    setAccount('')
    setContract(null)
    setProvider(null)
    setTransactions([])
  }

  // Obtener datos del contrato
  const getData = async () => {
    try {
      if (!contract) return null
      const data = await contract.getData()
      return data
    } catch (error) {
      console.error('Error obteniendo datos:', error)
      return null
    }
  }

  // Establecer datos en el contrato
  const setData = async (newData) => {
    try {
      if (!contract) throw new Error('Contrato no conectado')
      
      setIsLoading(true)
      const tx = await contract.setData(newData)
      await tx.wait()
      
      console.log('✅ Datos actualizados:', newData)
      return true
    } catch (error) {
      console.error('Error estableciendo datos:', error)
      alert('Error: ' + error.message)
      return false
    } finally {
      setIsLoading(false)
    }
  }

  // Almacenar transacción
  const storeTransaction = async (to, amount, message) => {
    try {
      if (!contract) throw new Error('Contrato no conectado')
      
      setIsLoading(true)
      const tx = await contract.storeTransaction(to, amount, message)
      await tx.wait()
      
      console.log('✅ Transacción almacenada')
      await loadTransactions() // Recargar transacciones
      return true
    } catch (error) {
      console.error('Error almacenando transacción:', error)
      alert('Error: ' + error.message)
      return false
    } finally {
      setIsLoading(false)
    }
  }

  // Cargar todas las transacciones
  const loadTransactions = async () => {
    try {
      if (!contract) return
      
      const allTransactions = await contract.getAllTransactions()
      const formattedTransactions = allTransactions.map((tx, index) => ({
        id: index,
        from: tx.from,
        to: tx.to,
        amount: tx.amount.toString(),
        message: tx.message,
        timestamp: new Date(Number(tx.timestamp) * 1000).toLocaleString()
      }))
      
      setTransactions(formattedTransactions)
    } catch (error) {
      console.error('Error cargando transacciones:', error)
    }
  }

  // Ejecutar acción genérica
  const executeAction = async (action) => {
    try {
      if (!contract) throw new Error('Contrato no conectado')
      
      setIsLoading(true)
      const tx = await contract.executeAction(action)
      const receipt = await tx.wait()
      
      console.log('✅ Acción ejecutada:', action)
      return receipt
    } catch (error) {
      console.error('Error ejecutando acción:', error)
      alert('Error: ' + error.message)
      return null
    } finally {
      setIsLoading(false)
    }
  }

  // Verificar si ya hay una conexión
  const checkConnection = async () => {
    try {
      if (window.ethereum) {
        const accounts = await window.ethereum.request({
          method: 'eth_accounts'
        })
        
        if (accounts.length > 0) {
          await connectWallet()
        }
      }
    } catch (error) {
      console.error('Error verificando conexión:', error)
    }
  }

  // Escuchar cambios de cuenta
  useEffect(() => {
    if (window.ethereum) {
      window.ethereum.on('accountsChanged', (accounts) => {
        if (accounts.length === 0) {
          disconnectWallet()
        } else {
          setAccount(accounts[0])
        }
      })

      window.ethereum.on('chainChanged', () => {
        window.location.reload()
      })
    }

    checkConnection()

    return () => {
      if (window.ethereum) {
        window.ethereum.removeAllListeners('accountsChanged')
        window.ethereum.removeAllListeners('chainChanged')
      }
    }
  }, [])

  // Cargar transacciones cuando el contrato esté listo
  useEffect(() => {
    if (contract) {
      loadTransactions()
    }
  }, [contract])

  const value = {
    account,
    contract,
    provider,
    isLoading,
    transactions,
    connectWallet,
    disconnectWallet,
    getData,
    setData,
    storeTransaction,
    loadTransactions,
    executeAction
  }

  return (
    <BlockchainContext.Provider value={value}>
      {children}
    </BlockchainContext.Provider>
  )
}
