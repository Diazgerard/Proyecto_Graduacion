// src/context/BlockchainContext.jsx
import { createContext, useContext, useState, useEffect } from "react";
import { ethers } from "ethers";
import contractABI from "../contracts/contractABI.json";
import contractAddress from "../contracts/contract-address.json";

export const BlockchainContext = createContext();

export const BlockchainProvider = ({ children }) => {
  const [account, setAccount] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [contract, setContract] = useState(null);
  const [provider, setProvider] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [allMessages, setAllMessages] = useState([]);
  const [message, setMessageState] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Conectar wallet
  const connectWallet = async () => {
    if (!window.ethereum) {
      alert("Instala MetaMask para continuar");
      return;
    }

    try {
      const [selectedAccount] = await window.ethereum.request({ method: "eth_requestAccounts" });
      setAccount(selectedAccount);
      setIsConnected(true);

      const ethersProvider = new ethers.BrowserProvider(window.ethereum);
      const signer = await ethersProvider.getSigner();
      const contractInstance = new ethers.Contract(contractAddress.address, contractABI.abi, signer);
      
      setProvider(ethersProvider);
      setContract(contractInstance);

      // Cargar mensaje inicial
      const msg = await contractInstance.getData();
      setMessageState(msg);

      // Cargar TODAS las transacciones iniciales (eventos del contrato)
      await loadAllTransactions(contractInstance, ethersProvider);
      
      console.log('Wallet conectada y datos cargados');

      // Escuchar eventos en tiempo real
      contractInstance.on("DataUpdated", async (sender, newData, timestamp, event) => {
        console.log("🔥 Nuevo evento recibido:", { sender, newData, timestamp });
        
        // Actualizar mensaje actual
        setMessageState(newData);
        
        // Crear nuevo registro de transacción
        try {
          const txRecord = {
            from: sender,
            message: newData,
            amount: newData, // Para compatibilidad
            date: new Date(Number(timestamp) * 1000).toLocaleString(),
            timestamp: new Date(Number(timestamp) * 1000),
            blockNumber: event.log.blockNumber,
            transactionHash: event.log.transactionHash
          };
          
          setTransactions(prev => [txRecord, ...prev]);
          setAllMessages(prev => [txRecord, ...prev]);
          
          console.log('✅ Nueva transacción agregada:', txRecord);
        } catch (err) {
          console.error("❌ Error procesando nuevo evento:", err);
        }
      });

    } catch (err) {
      console.error(err);
    }
  };

  const disconnectWallet = () => {
    // Limpiar listeners de eventos si existe el contrato
    if (contract) {
      contract.removeAllListeners("DataUpdated");
    }
    
    setAccount(null);
    setIsConnected(false);
    setContract(null);
    setProvider(null);
    setTransactions([]);
    setAllMessages([]);
    setMessageState("");
    setIsLoading(false);
    
    console.log('👋 Wallet desconectada');
  };

  // Escuchar cambios de cuenta
  useEffect(() => {
    if (window.ethereum) {
      window.ethereum.on("accountsChanged", (accounts) => {
        if (accounts.length > 0) setAccount(accounts[0]);
        else disconnectWallet();
      });
    }
  }, []);

  // Guardar mensaje en blockchain
  const setData = async (newMessage) => {
    if (!contract || !account) {
      throw new Error("Wallet no conectada o contrato no disponible");
    }

    try {
      setIsLoading(true);
      
      // Enviar transacción
      const tx = await contract.setData(newMessage);
      console.log("Transacción enviada:", tx.hash);
      
      // Esperar confirmación
      const receipt = await tx.wait();
      console.log("Transacción confirmada:", receipt);
      
      // El evento se procesará automáticamente por el listener
      return { success: true, hash: tx.hash };
      
    } catch (err) {
      console.error("Error enviando transacción:", err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Obtener mensaje actual
  const getData = async () => {
    if (!contract) return "";
    const msg = await contract.getData();
    setMessageState(msg);
    return msg;
  };

  // Obtener todas las transacciones desde los eventos
  const getTransactions = async () => {
    if (!contract) return [];
    try {
      const filter = contract.filters.DataUpdated();
      const events = await contract.queryFilter(filter);
      
      const transactions = await Promise.all(events.map(async (event) => {
        try {
          const block = await provider.getBlock(event.blockNumber);
          return {
            from: event.args.by || event.args.sender,
            amount: event.args.newData,
            date: new Date(block.timestamp * 1000).toLocaleString(),
            blockNumber: event.blockNumber,
            transactionHash: event.transactionHash,
            timestamp: block.timestamp
          };
        } catch (err) {
          console.error(`Error processing event:`, err);
          return null;
        }
      }));
      
      return transactions.filter(tx => tx !== null).reverse();
    } catch (err) {
      console.error("Error al obtener transacciones:", err);
      return transactions; // Retornar las transacciones locales si falla
    }
  };


  // Cargar TODAS las transacciones desde el inicio del contrato
  const loadAllTransactions = async (contractInstance, ethersProvider) => {
    try {
      console.log('🔄 Cargando todas las transacciones...');
      
      // Obtener todos los eventos DataUpdated desde el bloque 0
      const filter = contractInstance.filters.DataUpdated();
      const events = await contractInstance.queryFilter(filter, 0, 'latest');
      
      console.log(`📊 Encontrados ${events.length} eventos en total`);
      
      if (events.length === 0) {
        console.log('ℹ️  No hay eventos aún');
        setTransactions([]);
        setAllMessages([]);
        return;
      }
      
      // Procesar todos los eventos
      const processedTransactions = events.map((event) => {
        try {
          return {
            from: event.args.by,
            message: event.args.newData,
            amount: event.args.newData, // Para compatibilidad
            timestamp: new Date(Number(event.args.timestamp) * 1000),
            date: new Date(Number(event.args.timestamp) * 1000).toLocaleString(),
            blockNumber: event.blockNumber,
            transactionHash: event.transactionHash
          };
        } catch (err) {
          console.error(`❌ Error procesando evento:`, err);
          return null;
        }
      });
      
      // Filtrar eventos válidos y ordenar por más recientes primero
      const validTransactions = processedTransactions
        .filter(tx => tx !== null)
        .sort((a, b) => b.timestamp - a.timestamp);
      
      console.log(`✅ ${validTransactions.length} transacciones procesadas correctamente`);
      
      setTransactions(validTransactions);
      setAllMessages(validTransactions);
      
    } catch (err) {
      console.error("❌ Error cargando todas las transacciones:", err);
      setTransactions([]);
      setAllMessages([]);
    }
  };

  // Función para refrescar datos manualmente
  const refreshData = async () => {
    if (!contract || !provider) return;
    
    setIsLoading(true);
    try {
      await loadAllTransactions(contract, provider);
      const currentMessage = await contract.getData();
      setMessageState(currentMessage);
      console.log('🔄 Datos refrescados');
    } catch (err) {
      console.error("❌ Error refrescando datos:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <BlockchainContext.Provider
      value={{
        account,
        isConnected,
        connectWallet,
        disconnectWallet,
        contract,
        provider,
        message,
        setData,
        getData,
        transactions,
        allMessages,
        getTransactions,
        isLoading,
        refreshData,
      }}
    >
      {children}
    </BlockchainContext.Provider>
  );
};

// Hook para usar en componentes
export const useBlockchain = () => useContext(BlockchainContext);
