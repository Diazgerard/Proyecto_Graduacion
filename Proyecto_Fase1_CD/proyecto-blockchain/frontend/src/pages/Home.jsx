// src/pages/Home.jsx
import { useState } from "react";
import { useBlockchain } from "../context/BlockchainContext";
import ConnectWalletButton from "../components/ConnectWalletButton";

export default function Home() {
  const { isConnected, message, setData } = useBlockchain();
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    setLoading(true);
    setSuccessMessage("");
    try {
      await setData(input);
      setInput("");
      setSuccessMessage("¡Mensaje enviado exitosamente a la blockchain!");
      setTimeout(() => setSuccessMessage(""), 5000);
    } catch (err) {
      console.error("Error al enviar mensaje:", err);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Hero Section */}
      <section className="px-6 py-12">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-3xl md:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight">
            Almacena Datos en la
            <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent"> Blockchain</span>
          </h1>
          <p className="text-lg md:text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Una aplicación descentralizada que permite almacenar y gestionar datos de forma segura en la blockchain de Ethereum.
          </p>

          {/* Send Message Form */}
          {isConnected ? (
            <div className="max-w-2xl mx-auto mt-12">
              <div className="glass rounded-2xl p-8">
                <h2 className="text-2xl font-bold text-white mb-6">Enviar Mensaje a la Blockchain</h2>
                
                {successMessage && (
                  <div className="bg-green-500/20 border border-green-500/50 rounded-lg p-4 mb-6">
                    <p className="text-green-400 text-center">{successMessage}</p>
                  </div>
                )}
                
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div>
                    <label htmlFor="message" className="block text-gray-300 text-sm font-medium mb-3">
                      Tu Mensaje
                    </label>
                    <textarea
                      id="message"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Escribe tu mensaje que será almacenado permanentemente en la blockchain..."
                      className="w-full bg-slate-800/50 border border-slate-600 rounded-xl px-4 py-4 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-base"
                      rows={4}
                      required
                    />
                  </div>
                  
                  <button
                    type="submit"
                    disabled={loading || !input.trim()}
                    className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white py-4 px-6 rounded-xl font-medium transition-all duration-300 transform hover:scale-[1.02] flex items-center justify-center text-base"
                  >
                    {loading && (
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-3"></div>
                    )}
                    <span>{loading ? 'Enviando a la Blockchain...' : 'Enviar Mensaje'}</span>
                  </button>
                </form>

                {message && (
                  <div className="mt-6 p-4 bg-slate-800/50 rounded-xl">
                    <p className="text-gray-300 text-sm mb-1">Último mensaje almacenado:</p>
                    <p className="text-white font-medium">{message}</p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="max-w-2xl mx-auto mt-12">
              <div className="glass rounded-2xl p-8 text-center">
                <h2 className="text-2xl font-bold text-white mb-4">Conecta tu Wallet</h2>
                <p className="text-gray-300 mb-6">Para enviar mensajes a la blockchain, necesitas conectar tu wallet de Ethereum.</p>
                <ConnectWalletButton />
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Features Section */}
      <section className="px-6 py-16">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-2xl md:text-4xl font-bold text-white mb-4">
              Características Principales
            </h2>
            <p className="text-gray-300 text-lg max-w-2xl mx-auto">
              Descubre las potentes funcionalidades de nuestra aplicación blockchain
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="glass rounded-2xl p-8 transform hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center mb-6 mx-auto">
                <span className="text-2xl font-bold text-white">🔒</span>
              </div>
              <h3 className="text-xl font-bold text-white mb-4 text-center">Almacenamiento Seguro</h3>
              <p className="text-gray-300 text-center leading-relaxed">
                Tus datos se almacenan de forma inmutable y segura en la blockchain de Ethereum
              </p>
            </div>
            
            <div className="glass rounded-2xl p-8 transform hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-blue-600 rounded-xl flex items-center justify-center mb-6 mx-auto">
                <span className="text-2xl font-bold text-white">👁️</span>
              </div>
              <h3 className="text-xl font-bold text-white mb-4 text-center">Transparencia Total</h3>
              <p className="text-gray-300 text-center leading-relaxed">
                Todas las transacciones son públicas y verificables en la blockchain
              </p>
            </div>
            
            <div className="glass rounded-2xl p-8 transform hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl flex items-center justify-center mb-6 mx-auto">
                <span className="text-2xl font-bold text-white">🌐</span>
              </div>
              <h3 className="text-xl font-bold text-white mb-4 text-center">Descentralizado</h3>
              <p className="text-gray-300 text-center leading-relaxed">
                Sin intermediarios, conecta directamente con los smart contracts
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section className="px-6 py-16 bg-slate-900/50">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-2xl md:text-4xl font-bold text-white mb-4">
              ¿Cómo Funciona?
            </h2>
            <p className="text-gray-300 text-lg">
              Proceso simple en 3 pasos
            </p>
          </div>
          
          <div className="space-y-8">
            <div className="flex items-center space-x-6 p-6 glass rounded-2xl">
              <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold text-lg">1</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-white mb-2">Conecta tu Wallet</h3>
                <p className="text-gray-300">Conecta tu wallet de Ethereum (MetaMask) para interactuar con la blockchain</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-6 p-6 glass rounded-2xl">
              <div className="w-12 h-12 bg-purple-500 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold text-lg">2</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-white mb-2">Envía tu Mensaje</h3>
                <p className="text-gray-300">Escribe tu mensaje y envíalo al smart contract en la blockchain</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-6 p-6 glass rounded-2xl">
              <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-white font-bold text-lg">3</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-white mb-2">Visualiza y Gestiona</h3>
                <p className="text-gray-300">Ve tus datos almacenados y el historial de transacciones en el dashboard</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Tech Stack Section */}
      <section className="px-6 py-16">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-2xl md:text-4xl font-bold text-white mb-4">
            Stack Tecnológico
          </h2>
          <p className="text-gray-300 text-lg mb-12">
            Construido con las mejores tecnologías web3
          </p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="glass rounded-2xl p-6 transform hover:scale-105 transition-all duration-300">
              <div className="text-3xl mb-3">⚛️</div>
              <h3 className="text-white font-bold">React</h3>
              <p className="text-gray-400 text-sm mt-2">Frontend Framework</p>
            </div>
            
            <div className="glass rounded-2xl p-6 transform hover:scale-105 transition-all duration-300">
              <div className="text-3xl mb-3">⚡</div>
              <h3 className="text-white font-bold">Ethereum</h3>
              <p className="text-gray-400 text-sm mt-2">Blockchain</p>
            </div>
            
            <div className="glass rounded-2xl p-6 transform hover:scale-105 transition-all duration-300">
              <div className="text-3xl mb-3">🔗</div>
              <h3 className="text-white font-bold">Solidity</h3>
              <p className="text-gray-400 text-sm mt-2">Smart Contracts</p>
            </div>
            
            <div className="glass rounded-2xl p-6 transform hover:scale-105 transition-all duration-300">
              <div className="text-3xl mb-3">🎨</div>
              <h3 className="text-white font-bold">Tailwind</h3>
              <p className="text-gray-400 text-sm mt-2">Styling</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
