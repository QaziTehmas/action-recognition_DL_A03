import { useState } from 'react'
import './App.css'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedImage(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResult(null)
    }
  }

  const handleClear = () => {
    setSelectedImage(null)
    setPreviewUrl(null)
    setResult(null)
  }

  const handlePredict = async () => {
    if (!selectedImage) return

    setLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append('image', selectedImage)

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      // Check if response has the expected format
      if (!data.actions || !data.caption) {
        console.error("Unexpected response format:", data)
        alert("Backend returned an unexpected format. Please make sure you have restarted the backend server to apply the latest changes.")
        return
      }

      setResult(data)
    } catch (error) {
      console.error("Error predicting:", error)
      alert(error.message || "Failed to get prediction. Is the backend running?")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#0b0b0b] text-white font-sans flex flex-col p-4 box-border">
      <header className="text-center mb-4 shrink-0 h-[60px]">
        <h1 className="text-2xl font-semibold mb-1"> Action Recognition & Captioning</h1>
        <p className="text-[#a0a0a0] text-sm m-0">Upload an image to detect the action and generate a caption.</p>
      </header>

      <main className="flex flex-1 gap-4 min-h-0 w-full overflow-hidden max-md:flex-col max-md:overflow-y-auto">
        {/* Left Column: Image Upload & Preview - Fixed 50% width */}
        <div className="w-1/2 flex flex-col gap-4 h-full min-w-0 max-md:w-full max-md:flex-none max-md:min-h-[500px]">
          <div className="bg-[#1a1a1a] rounded-xl overflow-hidden flex-1 flex items-center justify-center border border-[#333] relative w-full min-h-0">
            {previewUrl ? (
              <div className="w-full h-full flex flex-col relative">
                <div className="px-4 py-2 bg-black/70 flex justify-between items-center absolute top-0 left-0 right-0 z-10 backdrop-blur-sm">
                  <span>image</span>
                  <button className="bg-none border-none text-white text-2xl cursor-pointer p-0 leading-none opacity-80 hover:opacity-100" onClick={handleClear}>×</button>
                </div>
                <img src={previewUrl} alt="Preview" className="w-full h-full object-contain bg-black" />
              </div>
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  id="file-upload"
                  className="hidden"
                />
                <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center justify-center gap-4 text-[#a0a0a0] transition-colors duration-200 w-full h-full hover:text-white hover:bg-white/5">
                  <div className="text-5xl bg-[#2a2a2a] w-20 h-20 rounded-full flex items-center justify-center mb-4">↑</div>
                  <span>Click to Upload Image</span>
                </label>
              </div>
            )}
          </div>

          <div className="grid grid-cols-[1fr_2fr] gap-4 shrink-0 h-[50px]">
            <button
              className="rounded-lg border-none font-semibold cursor-pointer transition-all duration-200 text-sm uppercase tracking-wide bg-[#333] text-[#ccc] hover:bg-[#444] hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handleClear}
              disabled={!previewUrl}
            >
              Clear
            </button>
            <button
              className="rounded-lg border-none font-semibold cursor-pointer transition-all duration-200 text-sm uppercase tracking-wide bg-[#e65100] text-white hover:bg-[#ff6d00] hover:-translate-y-px disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handlePredict}
              disabled={!selectedImage || loading}
            >
              {loading ? 'Analyzing...' : 'Submit'}
            </button>
          </div>
        </div>

        {/* Right Column: Results - Fixed 50% width */}
        <div className="w-1/2 flex flex-col gap-4 h-full min-w-0 max-md:w-full max-md:flex-none max-md:min-h-[500px]">
          {/* Predicted Actions Section */}
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] overflow-hidden flex flex-col flex-1 min-h-0">
            <div className="px-5 py-3 bg-white/5 border-b border-[#333] font-semibold flex items-center gap-3 shrink-0 text-base text-[#a0a0a0]">
              Predicted Actions
            </div>
            <div className="p-6 flex-1 overflow-y-auto flex flex-col justify-center">
              {result ? (
                <div className="w-full h-full flex flex-col justify-center">
                  <div className="text-center text-4xl font-extrabold mb-8 uppercase text-[#e65100] tracking-wide drop-shadow-[0_2px_10px_rgba(230,81,0,0.2)]">
                    {result.actions[0].label}
                  </div>
                  <div className="flex flex-col gap-4 w-full flex-1 justify-center">
                    {result.actions.map((action, index) => (
                      <div key={index} className="flex flex-col gap-1">
                        <div className="flex justify-between text-sm text-[#a0a0a0] font-medium">
                          <span>{action.label}</span>
                          <span>{Math.round(action.score * 100)}%</span>
                        </div>
                        <div className="h-2.5 bg-[#2a2a2a] rounded-full overflow-hidden">
                          <div
                            className="h-full bg-[#e65100] rounded-full transition-[width] duration-800 ease-[cubic-bezier(0.4,0,0.2,1)]"
                            style={{ width: `${action.score * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-[#a0a0a0] text-center italic opacity-50">
                  Predictions will appear here...
                </div>
              )}
            </div>
          </div>

          {/* Generated Caption Section */}
          <div className="bg-[#1a1a1a] rounded-xl border border-[#333] overflow-hidden flex flex-col flex-1 min-h-0">
            <div className="px-5 py-3 bg-white/5 border-b border-[#333] font-semibold flex items-center gap-3 shrink-0 text-base text-[#a0a0a0]">
              Generated Caption
            </div>
            <div className="p-6 flex-1 overflow-y-auto flex flex-col justify-center">
              {result ? (
                <div className="bg-[#2a2a2a] p-8 rounded-lg text-white leading-relaxed text-lg text-center border-l-4 border-[#e65100] my-auto">
                  {result.caption}
                </div>
              ) : (
                <div className="text-[#a0a0a0] text-center italic opacity-50">
                  Caption will appear here...
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
