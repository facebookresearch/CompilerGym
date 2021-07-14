import React, { useEffect, useState, useRef} from 'react'

const MIN_WIDTH = 170;

const LeftPanel = ({ children, leftWidth, setLeftWidth }) => {
    const leftRef = useRef()
    useEffect(() => {
      if (leftRef.current) {
        if (!leftWidth) {
          setLeftWidth(leftRef.current.clientWidth);
          return;
        }
        leftRef.current.style.width = `${leftWidth}px`;
      }
    }, [leftRef, leftWidth, setLeftWidth]);

    return <div ref={leftRef} className="leftPane">{children}</div>;
}

const PanelsContainer = ({left, right, className}) => {
    const [leftWidth, setLeftWidth] = useState(window.innerWidth/2);
    const [separatorXPosition, setSeparatorXPosition] = useState(undefined)
    const [dragging, setDragging] = useState(false);
    const splitPanelRef = useRef();

    const onMouseDown = (e) => {
      setSeparatorXPosition(e.clientX);
      setDragging(true);
    };

    const onTouchStart = (e) => {
      setSeparatorXPosition(e.touches[0].clientX);
      setDragging(true);
    };

    const onMove = (clientX) => {
      if (dragging && leftWidth && separatorXPosition) {
        const newLeftWidth = leftWidth + clientX - separatorXPosition;
        setSeparatorXPosition(clientX);

        if (newLeftWidth < MIN_WIDTH) {
          setLeftWidth(MIN_WIDTH);
          return;
        }

        if (splitPanelRef.current) {
          let splitPanelWidth = splitPanelRef.current.clientWidth;

          if (newLeftWidth > splitPanelWidth - MIN_WIDTH) {
            setLeftWidth(splitPanelWidth - MIN_WIDTH);
            return;
          }
        }

        setLeftWidth(newLeftWidth);
      }
    };

    const onMouseMove = (e) => {
      e.preventDefault();
      onMove(e.clientX);
    };

    const onTouchMove = (e) => {
      onMove(e.touches[0].clientX);
    };

    const onMouseUp = () => {
      setDragging(false);
    };

    useEffect(() => {
      if (dragging) {
        document.addEventListener("mousemove", onMouseMove);
        document.addEventListener("touchmove", onTouchMove);
        document.addEventListener("mouseup", onMouseUp);
      }
      return () => {
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("touchmove", onTouchMove);
        document.removeEventListener("mouseup", onMouseUp);
      };
    });


    return (
        <div ref={splitPanelRef} className={`splitView ${className || ""}`}>
          <LeftPanel leftWidth={leftWidth} setLeftWidth={setLeftWidth}>
            {left}
          </LeftPanel>

          <div
            className="divider-hitbox"
            onMouseDown={onMouseDown}
            onTouchStart={onTouchStart}
            onTouchEnd={onMouseUp}
          >
            <div className="divider" />
          </div>
          <div className="rightPane">{right}</div>
        </div>
    )
}

export default PanelsContainer;
