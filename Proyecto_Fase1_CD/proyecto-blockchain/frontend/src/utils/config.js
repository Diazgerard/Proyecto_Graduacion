// Configuración de redes
export const NETWORKS = {
  LOCALHOST: {
    chainId: '0x7a69', // 31337 en hexadecimal
    chainName: 'Localhost',
    rpcUrls: ['http://127.0.0.1:8545'],
    nativeCurrency: {
      name: 'Ethereum',
      symbol: 'ETH',
      decimals: 18
    }
  },
  SEPOLIA: {
    chainId: '0xaa36a7', // 11155111 en hexadecimal
    chainName: 'Sepolia testnet',
    rpcUrls: ['https://sepolia.infura.io/v3/'],
    nativeCurrency: {
      name: 'Ethereum',
      symbol: 'ETH',
      decimals: 18
    },
    blockExplorerUrls: ['https://sepolia.etherscan.io']
  }
}

// Red por defecto
export const DEFAULT_NETWORK = NETWORKS.LOCALHOST

// Chain IDs permitidos
export const ALLOWED_CHAIN_IDS = [31337, 11155111] // Localhost y Sepolia

// Configuración de la aplicación
export const APP_CONFIG = {
  name: 'Blockchain dApp',
  version: '1.0.0',
  description: 'Una dApp moderna con React y Hardhat',
  author: 'Tu Nombre',
  repository: 'https://github.com/tu-usuario/blockchain-dapp'
}

// Configuración de MetaMask
export const METAMASK_CONFIG = {
  method: 'eth_requestAccounts'
}

// Configuración de transacciones
export const TRANSACTION_CONFIG = {
  confirmations: 1,
  timeout: 60000, // 60 segundos
  gasLimit: '21000'
}

// URLs de servicios externos
export const EXTERNAL_URLS = {
  metamaskDownload: 'https://metamask.io/download/',
  hardhatDocs: 'https://hardhat.org/docs/',
  etherscanSepolia: 'https://sepolia.etherscan.io'
}

// Configuración de UI
export const UI_CONFIG = {
  theme: {
    colors: {
      primary: '#3B82F6',
      secondary: '#10B981',
      danger: '#EF4444',
      warning: '#F59E0B',
      info: '#06B6D4'
    }
  },
  animations: {
    duration: 200,
    easing: 'ease-in-out'
  }
}

// Mensajes de error comunes
export const ERROR_MESSAGES = {
  WALLET_NOT_CONNECTED: 'Wallet no conectada',
  WRONG_NETWORK: 'Red incorrecta. Por favor cambia a la red correcta.',
  TRANSACTION_FAILED: 'La transacción falló',
  CONTRACT_NOT_FOUND: 'No se pudo encontrar el contrato',
  INSUFFICIENT_FUNDS: 'Fondos insuficientes',
  USER_REJECTED: 'Transacción rechazada por el usuario',
  METAMASK_NOT_INSTALLED: 'MetaMask no está instalado'
}

// Configuración de desarrollo
export const DEV_CONFIG = {
  enableLogs: true,
  mockData: false,
  debugMode: process.env.NODE_ENV === 'development'
}
