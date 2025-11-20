import { Link } from 'react-router-dom'
import { useBlockchain } from '../context/BlockchainContext'
import ConnectWalletButton from '../components/ConnectWalletButton'

const Home = () => {
  const { account } = useBlockchain()

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-600/20 to-pink-600/20"></div>
        <div className="relative max-w-4xl mx-auto px-6 py-12">
          <div className="text-center animate-fadeInUp">
            <div className="inline-flex items-center bg-white/10 backdrop-blur-md rounded-full px-4 py-2 mb-6 text-sm">
              <span className="text-white font-medium">Blockchain Ready</span>
            </div>
            
            <h1 className="text-3xl md:text-5xl font-bold bg-gradient-to-r from-white via-purple-200 to-pink-200 bg-clip-text text-transparent mb-6 leading-tight">
              CryptoHub dApp
            </h1>
            
            <p className="text-lg text-gray-200 mb-8 max-w-2xl mx-auto leading-relaxed">
              Explora el futuro de las aplicaciones descentralizadas. 
              Conecta tu wallet y descubre el poder de la blockchain.
            </p>
            
            {!account ? (
              <div className="space-y-4">
                <ConnectWalletButton />
                <div className="flex items-center justify-center text-gray-300">
                  <span className="text-sm">Requiere MetaMask para comenzar</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="inline-flex items-center glass px-6 py-3 rounded-lg">
                  <span className="text-white font-medium text-sm">Wallet Conectada</span>
                </div>
                <Link
                  to="/dashboard"
                  className="inline-flex items-center bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-8 py-4 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 shadow-lg shadow-purple-500/25 text-lg"
                >
                  <span>Acceder al Dashboard</span>
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-6xl mx-auto px-6 py-16">
        <div className="text-center mb-12">
          <h2 className="text-2xl md:text-3xl font-bold text-white mb-4">
            Funcionalidades Principales
          </h2>
          <p className="text-gray-300 text-lg max-w-2xl mx-auto">
            Descubre todo lo que puedes hacer con nuestra plataforma blockchain
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          <div className="glass rounded-xl p-8 text-center hover:scale-105 transition-all duration-300 group">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-400 to-purple-500 rounded-xl mx-auto mb-6 group-hover:scale-110 transition-transform"></div>
            <h3 className="text-xl font-semibold text-white mb-4">Conectar Wallet</h3>
            <p className="text-gray-300 text-base leading-relaxed">
              Integración perfecta con MetaMask para acceder a todas las funcionalidades de la dApp de forma segura.
            </p>
          </div>

          <div className="glass rounded-xl p-8 text-center hover:scale-105 transition-all duration-300 group">
            <div className="w-16 h-16 bg-gradient-to-r from-purple-400 to-pink-500 rounded-xl mx-auto mb-6 group-hover:scale-110 transition-transform"></div>
            <h3 className="text-xl font-semibold text-white mb-4">Dashboard Avanzado</h3>
            <p className="text-gray-300 text-base leading-relaxed">
              Interfaz intuitiva para interactuar con contratos inteligentes y gestionar tus datos blockchain.
            </p>
          </div>

          <div className="glass rounded-xl p-8 text-center hover:scale-105 transition-all duration-300 group">
            <div className="w-16 h-16 bg-gradient-to-r from-pink-400 to-red-500 rounded-xl mx-auto mb-6 group-hover:scale-110 transition-transform"></div>
            <h3 className="text-xl font-semibold text-white mb-4">Historial Completo</h3>
            <p className="text-gray-300 text-base leading-relaxed">
              Visualiza todas las transacciones y operaciones registradas en la blockchain de forma transparente.
            </p>
          </div>
        </div>
      </div>

      {/* How it Works Section */}
      <div className="max-w-4xl mx-auto px-6 py-16">
        <div className="glass rounded-2xl p-8 md:p-12">
          <h2 className="text-2xl md:text-3xl font-bold text-white mb-8 text-center">¿Cómo Funciona?</h2>
          <div className="space-y-8">
            <div className="flex items-start space-x-6">
              <div className="w-12 h-12 bg-gradient-to-r from-purple-400 to-pink-400 rounded-xl flex items-center justify-center text-white font-bold flex-shrink-0 text-lg">
                1
              </div>
              <div>
                <h4 className="text-xl font-semibold text-white mb-3">Gestión de Datos</h4>
                <p className="text-gray-300 text-base leading-relaxed">
                  Almacena y recupera información directamente en la blockchain usando contratos inteligentes seguros y verificables.
                </p>
              </div>
            </div>
            
            <div className="flex items-start space-x-6">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-400 to-purple-400 rounded-xl flex items-center justify-center text-white font-bold flex-shrink-0 text-lg">
                2
              </div>
              <div>
                <h4 className="text-xl font-semibold text-white mb-3">Transacciones Seguras</h4>
                <p className="text-gray-300 text-base leading-relaxed">
                  Crea y registra transacciones personalizadas con total transparencia e inmutabilidad en la red Ethereum.
                </p>
              </div>
            </div>
            
            <div className="flex items-start space-x-6">
              <div className="w-12 h-12 bg-gradient-to-r from-pink-400 to-red-400 rounded-xl flex items-center justify-center text-white font-bold flex-shrink-0 text-lg">
                3
              </div>
              <div>
                <h4 className="text-xl font-semibold text-white mb-3">Ejecución de Acciones</h4>
                <p className="text-gray-300 text-base leading-relaxed">
                  Interactúa con funciones avanzadas del contrato de manera segura y eficiente usando tu wallet conectada.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="max-w-4xl mx-auto px-6 py-16">
        <div className="text-center">
          <h3 className="text-2xl md:text-3xl font-bold text-white mb-8">Tecnologías de Vanguardia</h3>
          <div className="flex flex-wrap justify-center items-center gap-6">
            {[
              { name: 'React', color: 'from-blue-400 to-cyan-400' },
              { name: 'Hardhat', color: 'from-yellow-400 to-orange-400' },
              { name: 'MetaMask', color: 'from-orange-400 to-red-400' },
              { name: 'Ethereum', color: 'from-purple-400 to-indigo-400' },
              { name: 'Tailwind', color: 'from-cyan-400 to-blue-400' }
            ].map((tech) => (
              <div key={tech.name} className="glass rounded-xl px-6 py-4 hover:scale-105 transition-all duration-300">
                <div className="flex items-center space-x-3">
                  <div className={`w-12 h-12 bg-gradient-to-r ${tech.color} rounded-xl`}></div>
                  <span className="text-white font-medium text-lg">{tech.name}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Home
