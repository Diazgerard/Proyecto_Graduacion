// Formatear direcciones de Ethereum
export const formatAddress = (address) => {
  if (!address) return ''
  return `${address.slice(0, 6)}...${address.slice(-4)}`
}

// Formatear números con decimales
export const formatNumber = (number, decimals = 2) => {
  if (!number) return '0'
  return Number(number).toFixed(decimals)
}

// Formatear timestamps a fecha legible
export const formatTimestamp = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(Number(timestamp) * 1000)
  return date.toLocaleString()
}

// Formatear cantidades en ETH
export const formatEther = (amount) => {
  if (!amount) return '0 ETH'
  return `${formatNumber(amount)} ETH`
}

// Validar dirección de Ethereum
export const isValidAddress = (address) => {
  return /^0x[a-fA-F0-9]{40}$/.test(address)
}

// Copiar texto al portapapeles
export const copyToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch (error) {
    console.error('Error copiando al portapapeles:', error)
    return false
  }
}

// Truncar texto largo
export const truncateText = (text, maxLength = 50) => {
  if (!text) return ''
  if (text.length <= maxLength) return text
  return `${text.slice(0, maxLength)}...`
}

// Generar ID único
export const generateId = () => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2)
}

// Formatear tiempo transcurrido
export const timeAgo = (timestamp) => {
  const now = Date.now()
  const diff = now - (Number(timestamp) * 1000)
  
  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)
  
  if (days > 0) return `${days} día${days > 1 ? 's' : ''} atrás`
  if (hours > 0) return `${hours} hora${hours > 1 ? 's' : ''} atrás`
  if (minutes > 0) return `${minutes} minuto${minutes > 1 ? 's' : ''} atrás`
  return 'Hace un momento'
}

// Validar entrada de número
export const validateNumber = (value) => {
  const num = parseFloat(value)
  return !isNaN(num) && num >= 0
}

// Formatear estado de transacción
export const getTransactionStatus = (confirmed) => {
  return confirmed ? 
    { text: 'Confirmada', color: 'green', icon: '✅' } :
    { text: 'Pendiente', color: 'yellow', icon: '⏳' }
}
